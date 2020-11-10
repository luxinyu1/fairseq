#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

import numpy as np
import torch

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from omegaconf import DictConfig
from fairseq.trainer import Trainer


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(cfg: DictConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    assert cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None, \
        'Must specify batch size either with --max-tokens or --batch-size'
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in cfg.dataset.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {} ({})".format(cfg.task._name, task.__class__.__name__))
    logger.info("model: {} ({})".format(cfg.model._name, model.__class__.__name__))
    logger.info(
        "criterion: {} ({})".format(cfg.criterion._name, criterion.__class__.__name__)
    )
    logger.info("num. model params: {} (num. trained: {})".format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)

    logger.info('training on {} devices (GPUs/TPUs)'.format(cfg.distributed_training.distributed_world_size))
    logger.info('max tokens per GPU = {} and batch size per GPU = {}'.format(
        cfg.dataset.max_tokens,
        cfg.dataset.batch_size,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while (
        lr > cfg.optimization.min_lr
        and epoch_itr.next_epoch_idx <= max_epoch
    ):
        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(cfg.checkpoint.patience))
            return True
        else:
            return False


@metrics.aggregate("train")
def train(cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if getattr(cfg.common, "tpu", False):
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir if distributed_utils.is_master(cfg.distributed_training) else None
        ),
        default_log_format=('tqdm' if not cfg.common.no_progress_bar else 'simple'),
    )

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(',')
    should_stop = False
    num_updates = trainer.get_num_updates()
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr, valid_subsets: List[str], end_of_epoch: bool) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf
    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or num_updates >= max_update
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or num_updates >= max_update
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
        )
    ) and not cfg.dataset.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        # valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)
        valid_losses = sari_validate(cfg, trainer, task, epoch_itr, valid_subsets)
        sari = -valid_losses[0]

    # Stopping conditions
    should_stop = (
        should_stop_early(cfg, valid_losses[0])
        or num_updates >= max_update
        or (
            cfg.optimization.stop_time_hours > 0
            and trainer.cumulative_training_time() / (60 * 60) > cfg.optimization.stop_time_hours
        )
    )

    # Save checkpoint
    if do_save or should_stop:
        logger.info("begin save checkpoint")
        if sari > 35:
            checkpoint_utils.save_checkpoint(cfg.checkpoint, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop

def sari_validate(cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr, subsets: List[str]) -> List[Optional[float]]:
    from pathlib import Path
    from access.resources.paths import get_data_filepath
    from access.utils.helpers import read_lines
    from access.preprocessors import load_preprocessors, ComposedPreprocessor
    from easse.report import get_all_scores
    from fairseq.data import encoders
    from fairseq_cli.interactive import buffered_read, make_batches
    from fairseq_cli.generate import get_symbols_to_strip_from_output
    from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
    import tempfile

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(cfg.task)

    # TODO: Choose parameters for the preprocessors ?
    # 从pickle文件读取preprocessor
    # preprocessors = load_preprocessors(Path(cfg.task.data).parent)
    # composed_preprocessor = ComposedPreprocessor(preprocessors)
    # 获得turkcorpus.valid.complex的路径
    complex_filepath = get_data_filepath('turkcorpus', 'valid', 'complex')
    # make temp dir
    # encoded_complex_filepath = tempfile.mkstemp()[1]
    # encoded_pred_filepath = tempfile.mkstemp()[1]
    pred_filepath = tempfile.mkstemp()[1]
    # use preprocessors to encode complex file
    # composed_preprocessor.encode_file(complex_filepath, encoded_complex_filepath)
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        trainer.get_model().max_positions(),
    )
    parser = options.get_generation_parser(interactive=True)
    # TODO: Take args from fairseq_generate
    gen_args = options.parse_args_and_arch(parser, input_args=['/dummy_data', '--beam', '2'])
    # Initialize generator
    generator = task.build_generator([trainer.model], gen_args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(cfg.tokenizer)
    bpe = encoders.build_bpe(cfg.bpe)

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    with open(pred_filepath, 'w') as f:
        start_id = 0
        for inputs in buffered_read(complex_filepath, buffer_size=9999):
            results = []
            for batch in make_batches(inputs, cfg, task, max_positions, encode_fn):
                bsz = batch.src_tokens.size(0)
                src_tokens = batch.src_tokens
                src_lengths = batch.src_lengths
                constraints = batch.constraints
                if use_cuda:
                    src_tokens = src_tokens.cuda()
                    src_lengths = src_lengths.cuda()
                    if constraints is not None:
                        constraints = constraints.cuda()
                sample = {
                    "net_input": {
                        "src_tokens": src_tokens,
                        "src_lengths": src_lengths,
                    },
                }
                translations = task.inference_step(
                    generator, [trainer.model], sample, constraints=constraints
                )
                list_constraints = [[] for _ in range(bsz)]
                if cfg.generation.constraints:
                    list_constraints = [unpack_constraints(c) for c in constraints]
                for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                    src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                    constraints = list_constraints[i]
                    results.append(
                        (
                            start_id + id,
                            src_tokens_i,
                            hypos,
                            {
                                "constraints": constraints,
                            },
                        )
                    )

            # sort output to match input order
            for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                    for constraint in info["constraints"]:
                        pass

                # Process top predictions
                for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=src_str,
                        alignment=hypo["alignment"],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=cfg.common_eval.post_process,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                    )
                    detok_hypo_str = decode_fn(hypo_str)
                    # detokenized hypothesis
                    f.write(f'{detok_hypo_str}\n')
                    if cfg.generation.print_alignment:
                        alignment_str = " ".join(
                            ["{}-{}".format(src, tgt) for src, tgt in alignment]
                        )

            # update running id_ counter
            start_id += len(inputs)

        # composed_preprocessor.decode_file(encoded_pred_filepath, pred_filepath)
        ref_filepaths = [get_data_filepath('turkcorpus', 'valid', 'simple.turk', i) for i in range(8)]
        scores = get_all_scores(read_lines(complex_filepath), read_lines(pred_filepath), [read_lines(ref_filepath) for ref_filepath in ref_filepaths])
        print(f'num_updates={trainer.get_num_updates()}')
        print(f'ts_scores={scores}')
        sari = scores['SARI']
        if not hasattr(trainer, 'best_sari'):
            trainer.best_sari = 0
        if not hasattr(trainer, 'n_validations_since_best'):
            trainer.n_validations_since_best = 0
        if sari > trainer.best_sari:
            trainer.best_sari = sari
            trainer.n_validations_since_best = 0
        else:
            trainer.n_validations_since_best += 1
            print(f'SARI did not improve for {trainer.n_validations_since_best} validations')
            # Does not work because scheduler will set it to previous value everytime
            # trainer.optimizer.set_lr(0.75 * trainer.optimizer.get_lr())
            if trainer.n_validations_since_best >= cfg.validations_before_sari_early_stopping:
                print(f'Early stopping because SARI did not improve for {trainer.n_validations_since_best} validations')
                trainer.early_stopping = True

            def is_abort(epoch_itr, best_sari):
                if (epoch_itr.epoch >= 2 and best_sari < 19):
                    return True
                if (epoch_itr.epoch >= 5 and best_sari < 22):
                    return True
                if (epoch_itr.epoch >= 10 and best_sari < 25):
                    return True
                return False
            # if is_abort(epoch_itr, best_sari):
            #     print(f'Early stopping because best SARI is too low ({best_sari:.2f}) after {epoch_itr.epoch} epochs.')
            #     # Remove the checkpoint directory as we got nothing interesting
            #     shutil.rmtree(args.save_dir)
            #     # TODO: Abort
    return [-sari]

def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr, subsets: List[str]) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir if distributed_utils.is_master(cfg.distributed_training) else None
            ),
            default_log_format=('tqdm' if not cfg.common.no_progress_bar else 'simple'),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best, stats[cfg.checkpoint.best_checkpoint_metric]
        )
    return stats


def cli_main(modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == '__main__':
    cli_main()
