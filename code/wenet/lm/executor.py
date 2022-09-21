# Copyright 2019 Mobvoi Inc.
#           NPU, ASLP Group (Author: Qijie Shao)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import torch
from contextlib import nullcontext
from torch.nn.utils import clip_grad_norm_
import numpy as np
   
class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, args, scaler):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        print('using accumulate grad, new batch size is {} times'
                     'larger than before'.format(accum_grad))
        if use_amp:
            assert scaler is not None
        num_total_batch = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            key, text, lengths = batch
            text = text.to(device)
            lengths = lengths.to(device)
            
            context = None
            # Disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module
            # variables, which will later be synchronized.
            if is_distributed and batch_idx % accum_grad != 0:
                context = model.no_sync
            # Used for single gpu training and DDP gradient synchronization
            # processes.
            else:
                context = nullcontext
            with context():
                # autocast context
                # The more details about amp can be found in
                # https://pytorch.org/docs/stable/notes/amp_examples.html
                with torch.cuda.amp.autocast(scaler is not None):
                    loss = model(
                            text,
                            lengths,
                            )
                    loss = loss / accum_grad
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if batch_idx % accum_grad == 0:
                # Use mixed precision training
                if use_amp:
                    scaler.unscale_(optimizer)
                    grad_norm = clip_grad_norm_(model.parameters(), clip)
                    # Must invoke scaler.update() if unscale_() is used in the
                    # iteration to avoid the following error:
                    #   RuntimeError: unscale_() has already been called
                    #   on this optimizer since the last update().
                    # We don't check grad here since that if the gradient has
                    # inf/nan values, scaler.step will skip optimizer.step().
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = clip_grad_norm_(model.parameters(), clip)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                self.step += 1

            if batch_idx % log_interval == 0:
                log_str = 'TRAIN Batch {}/{} loss:{:.2f}'.format(batch_idx, num_total_batch, loss.item())
                logging.info(log_str)

    def cv(self, model, data_loader, device):
        ''' Cross validation
        '''
        model.eval()
        num_seen_utts = 1
        total_loss = 0.0
        num_total_batch = len(data_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, text, lengths = batch
                text = text.to(device)
                lengths = lengths.to(device)
                num_utts = lengths.size(0)
                loss = model(text, lengths)

                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts

        return total_loss/num_seen_utts
