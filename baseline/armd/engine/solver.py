import os
import sys
import time
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
try:
    from ema_pytorch import EMA
except ModuleNotFoundError:
    import copy

    class EMA:
        def __init__(self, model, beta=0.995, update_every=10):
            self.model = model
            self.ema_model = copy.deepcopy(model)
            self.beta = beta
            self.update_every = max(1, int(update_every))
            self.step = 0
            for param in self.ema_model.parameters():
                param.requires_grad_(False)

        def to(self, device):
            self.ema_model.to(device)
            return self

        def update(self):
            self.step += 1
            if self.step % self.update_every != 0:
                return
            with torch.no_grad():
                for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                    ema_param.data.mul_(self.beta).add_(param.data, alpha=1.0 - self.beta)
                for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers()):
                    ema_buffer.copy_(buffer)

        def state_dict(self):
            return {
                'ema_model': self.ema_model.state_dict(),
                'step': self.step,
            }

        def load_state_dict(self, state_dict):
            if 'ema_model' in state_dict:
                self.ema_model.load_state_dict(state_dict['ema_model'])
                self.step = state_dict.get('step', self.step)
            else:
                self.ema_model.load_state_dict(state_dict)
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from efficiency import benchmark_model_call, write_measurement


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        weight_decay = config['solver'].get('weight_decay', 0.0)
        self.opt = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=start_lr,
            betas=[0.9, 0.96],
            weight_decay=weight_decay,
        )
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def set_dataloader(self, dataloader):
        """Switch the round-robin trainer to the current city's loader."""
        self.dl = cycle(dataloader)

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def train(self, num_steps=None):
        device = self.device
        step = 0
        train_num_steps = self.train_num_steps if num_steps is None else int(num_steps)
        if train_num_steps < 1:
            raise ValueError('num_steps must be positive')
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=train_num_steps) as pbar:
            while step < train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num, size_every, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def sample_forecast(self, raw_dataloader, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        timestamps = np.empty([0, shape[0]])

        for idx, batch in enumerate(raw_dataloader):
            if len(batch)==3:
                x, t_m, t_stamp = batch
                x, t_m = x.to(self.device), t_m.to(self.device)
            elif len(batch)==2:
                x, t_m = batch
                x, t_m = x.to(self.device), t_m.to(self.device)
                t_stamp = None
            else:
                x = batch
                x = x.to(self.device)
                t_stamp = None
            efficiency_dir = getattr(self.args, 'efficiency_record_dir', '')
            if efficiency_dir and not getattr(self, '_efficiency_recorded', False):
                target_count = self.args.efficiency_batch_size
                indices = torch.arange(target_count, device=x.device) % x.shape[0]
                benchmark_x = x.index_select(0, indices)
                latency = benchmark_model_call(
                    lambda: self.ema.ema_model.generate_mts(benchmark_x),
                    sample_count=target_count,
                    warmup_calls=self.args.efficiency_warmup_calls,
                    measured_calls=self.args.efficiency_measured_calls,
                )
                write_measurement(
                    os.path.join(efficiency_dir, 'inference_armd.json'),
                    kind='inference', method='ARMD', stage='inference',
                    value=latency, unit='ms_per_sample',
                    note='model execution only; excludes loading, scaling, metrics, and file I/O',
                    metadata={
                        'batch_size': target_count,
                        'warmup_calls': self.args.efficiency_warmup_calls,
                        'measured_calls': self.args.efficiency_measured_calls,
                        'timing_protocol': 'independent_repetitions_mean',
                    },
                )
                self._efficiency_recorded = True
            sample = self.ema.ema_model.generate_mts(x)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x[:,shape[0]:,:].detach().cpu().numpy()])
            if t_stamp is not None:
                timestamps = np.row_stack([timestamps, t_stamp[:, shape[0]:].detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        if timestamps.shape[0] == 0:
            timestamps = None
        return samples, reals, timestamps
