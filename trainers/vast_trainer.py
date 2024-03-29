from .base_trainer import *


class VastTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    
    def train(self):
        # get training config
        train_cfg = self.config.trainer

        # build model, data, optimizer, and loss
        self.build_model()
        self.build_data()
        self.build_optimizer()
        self.build_loss()

        # prepare things
        best_val_loss = float('inf')
        os.makedirs(train_cfg.ckpt_dir, exist_ok=True)
        self.save_training_info(train_cfg.ckpt_dir)
        self.best_model = None
        
        # train loop
        for ep in range(train_cfg.max_epochs):
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                imgs = batch['image']
                pred = self.model(imgs, labels=batch)
                loss = self.loss(pred, batch)
                total_loss, html_loss, coord_loss, visual_align_loss = loss['total_loss'], loss['html_loss'], loss['coord_loss'], loss['visual_align_loss']
                # gradient descent
                self.optimizer.zero_grad()  # need to zero the gradients because pytorch accumulates them by default
                total_loss.backward()  # compute gradients of the loss w.r.t. the parameters
                self.optimizer.step()  # update the parameters
                self.lr_scheduler.step()  # update the learning rate

                current_lr = self.optimizer.param_groups[0]['lr']
                if i % train_cfg.log_interval == 0:
                    print(f"Epoch {ep}, Batch {i}, LR {current_lr:.4f},  \
                          Total Loss {total_loss.cpu().item():.3f},  \
                          Structure Loss {html_loss.cpu().item():.3f},   \
                          Coord Loss {coord_loss.cpu().item():.3f}, \
                          Visual Align Loss {visual_align_loss.cpu().item():.3f}")
            
            # validation loop
            self.model.eval()
            val_loss = 0
            for i, batch in enumerate(self.val_loader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                imgs = batch['image']
                with torch.no_grad():
                    pred = self.model(imgs, labels=None)
                loss = self.loss(pred, batch)
                total_loss = loss['total_loss']
                val_loss += total_loss.cpu().item()
            val_loss = val_loss / len(self.val_loader)
            print(f"Epoch {ep}, Val Loss {val_loss:.3f}")        

            # save checkpoint
            if val_loss < best_val_loss:
                if best_val_loss != float('inf'):
                    os.remove(os.path.join(train_cfg.ckpt_dir, f'best_model-val_loss={best_val_loss:.3f}.pt'))
                best_val_loss = val_loss
                self.best_model = deepcopy(self.model)
                torch.save(
                    self.model.state_dict(), 
                    os.path.join(train_cfg.ckpt_dir, f'best_model-val_loss={val_loss:.3f}.pt')
                )

