import yaml
import torch
import torch.nn as nn
import argparse
import pprint
import os

from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Generator, Discriminator, Vgg19
from dataset import IllustDataset
from loss import SCFTLossCalculator
from visualize import Visualizer
from utils import session
import contextlib

class Trainer:
    def __init__(self,
                 config,
                 outdir,
                 modeldir,
                 data_path,
                 sketch_path,
                 iteration):

        self.train_config = config["train"]
        self.data_config = config["dataset"]
        model_config = config["model"]
        self.loss_config = config["loss"]

        self.outdir = outdir
        self.modeldir = modeldir
        self.dataset = IllustDataset(data_path,
                                     sketch_path,
                                     self.data_config["line_method"],
                                     self.data_config["extension"],
                                     self.data_config["train_image_size"],
                                     self.data_config["valid_image_size"],
                                     self.data_config["color_space"],
                                     self.data_config["line_space"],
                                     self.data_config["src_perturbation"],
                                     self.data_config["tgt_perturbation"])
        print(self.dataset)

        if (iteration == 0):
          gen = Generator()
          self.gen, self.gen_opt = self._setting_model_optim(gen,
                                                            model_config["generator"])

          dis = Discriminator()
          self.dis, self.dis_opt = self._setting_model_optim(dis,
                                                            model_config["discriminator"])

        if (iteration>0): #continue training

            print("continuing training at " + str(iteration) + " iteration. Loading models...")

            self.gen = Generator().cuda()
            self.gen.load_state_dict(torch.load(f"{self.modeldir}/generator_{iteration}.pt"))
            
            self.gen_opt = torch.optim.Adam(self.gen.parameters())
            self.gen_opt.load_state_dict(torch.load(f"{self.modeldir}/gen_optimizer_{iteration}.pt"))
            
            self.dis = Discriminator().cuda()
            self.dis.load_state_dict(torch.load(f"{self.modeldir}/discriminator_{iteration}.pt"))
            
            self.dis_opt = torch.optim.Adam(self.dis.parameters())
            self.dis_opt.load_state_dict(torch.load(f"{self.modeldir}/dis_optimizer_{iteration}.pt"))

        self.iteration = iteration
        self.vgg = Vgg19(requires_grad=False)
        self.vgg.cuda()
        self.vgg.eval()

        self.lossfunc = SCFTLossCalculator()
        self.visualizer = Visualizer(self.data_config["color_space"])

    @staticmethod
    def _setting_model_optim(model: nn.Module,
                             config: Dict):
        model.cuda()
        if config["mode"] == "train":
            model.train()
        elif config["mode"] == "eval":
            model.eval()

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config["lr"],
                                     betas=(config["b1"], config["b2"]))

        return model, optimizer

    @staticmethod
    def _valid_prepare(dataset,
                       validsize: int) -> List[torch.Tensor]:

        c_val, l_val = dataset.valid(validsize)

        return [l_val, c_val]

    @staticmethod
    def _build_dict(loss_dict: Dict[str, float],
                    epoch: int,
                    num_epochs: int) -> Dict[str, str]:

        report_dict = {}
        report_dict["epoch"] = f"{epoch}/{num_epochs}"
        
        for k, v in loss_dict.items():
            report_dict[k] = f"{v:.4f}"
        
        return report_dict

    def _eval(self,
              iteration: int,
              validsize: int,
              v_list: List[torch.Tensor]):
        print()
        print("saving generator")
        torch.save(self.gen.state_dict(),
                   f"{self.modeldir}/generator_{iteration}.pt")
        print("saving discriminator")
        torch.save(self.dis.state_dict(),
                   f"{self.modeldir}/discriminator_{iteration}.pt")
        print("saving generator optimizer")
        torch.save(self.gen_opt.state_dict(),
                   f"{self.modeldir}/gen_optimizer_{iteration}.pt")
        print("saving discriminator optimizer")
        torch.save(self.dis_opt.state_dict(),
                   f"{self.modeldir}/dis_optimizer_{iteration}.pt")

        #just to save space 
        if (iteration > 1):
          print("removing previous snapshot models")

        with contextlib.suppress(FileNotFoundError):
            #keep the two most recent models just cause I'm curious
            previousFileIteration = iteration - self.train_config["snapshot_interval"]
            os.remove(f"{self.modeldir}/generator_{previousFileIteration}.pt")
            os.remove(f"{self.modeldir}/discriminator_{previousFileIteration}.pt")
            os.remove(f"{self.modeldir}/gen_optimizer_{previousFileIteration}.pt")
            os.remove(f"{self.modeldir}/dis_optimizer_{previousFileIteration}.pt")

        with torch.no_grad():
            y = self.gen(v_list[0], v_list[1])

        self.visualizer(v_list, y,
                        self.outdir, iteration, validsize)

    def _iter(self, data):
        jit, war, line = data
        jit = jit.cuda()
        war = war.cuda()
        line = line.cuda()

        loss = {}

        # Discriminator update
        y = self.gen(line, war)
        dis_loss = self.loss_config["adv"] * self.lossfunc.adversarial_disloss(self.dis,
                                                                               y.detach(),
                                                                               jit)

        self.dis_opt.zero_grad()
        dis_loss.backward()
        self.dis_opt.step()

        # Generator update
        y = self.gen(line, war)
        adv_gen_loss = self.loss_config["adv"] * self.lossfunc.adversarial_genloss(self.dis,
                                                                                   y)
        con_loss = self.loss_config["content"] * self.lossfunc.content_loss(y, jit)
        perceptual_loss, style_loss = self.lossfunc.style_and_perceptual_loss(self.vgg, y, jit)
        perceptual_loss = self.loss_config["perceptual"] * perceptual_loss
        style_loss = self.loss_config["style"] * style_loss

        gen_loss = adv_gen_loss + con_loss + perceptual_loss + style_loss

        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()

        loss["loss_adv_dis"] = dis_loss.item()
        loss["loss_adv_gen"] = adv_gen_loss.item()
        loss["loss_content"] = con_loss.item()
        loss["loss_perceptual"] = perceptual_loss.item()
        loss["loss_style"] = style_loss.item()
        return loss

    def __call__(self):
        iteration = self.iteration
        # starting_epoch = 0 if iteration == 0 else iteration / (self.data_config["validsize"]/ self.train_config["batchsize"])
        v_list = self._valid_prepare(self.dataset,
                                     self.train_config["validsize"],
                                     )
    
        for epoch in range(0, self.train_config["epoch"]):
            dataloader = DataLoader(self.dataset,
                                    batch_size=self.train_config["batchsize"],
                                    shuffle=True,
                                    drop_last=True)

            with tqdm(total=len(self.dataset)) as pbar:
                for index, data in enumerate(dataloader):
                    loss_dict = self._iter(data)
                    report_dict = self._build_dict(loss_dict,
                                                   epoch,
                                                   self.train_config["epoch"])

                    pbar.update(self.train_config["batchsize"])
                    pbar.set_postfix(**report_dict)

                    if iteration % self.train_config["snapshot_interval"] == 1:
                        self._eval(iteration,
                                   self.train_config["validsize"],
                                   v_list,
                                   )
                    iteration += 1
                    with open(self.modeldir/'loss.txt','a') as f:
                        f.write("iteration " + str(iteration) + "| ")
                        for k, v in loss_dict.items():
                            f.write(k+": "+ str(report_dict[k]) + "    ")
                        f.write("\n")
                    f.close()
            print("Epoch: ", epoch, "       iteration: ", iteration)
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCFT")
    parser.add_argument('--session', type=str, default='scft', help="session name")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    parser.add_argument('--sketchKeras_path', type=Path, help="path containing sketchKeras images")
    parser.add_argument('--digital_path', type=Path, help="path containing digital images")
    parser.add_argument('--iteration_count', type=int, default=0, help='the starting iteration count')
    parser.add_argument('--session_dir', type=Path, help='if we need to pull previously trained models')

    args = parser.parse_args()

    if (args.iteration_count) == 0:
        outdir, modeldir = session(args.session)
    elif (args.iteration_count) > 0:
        modeldir = args.session_dir / "ckpts"
        outdir = args.session_dir / "vis"
    with open("param.yaml", "r") as f:
        config = yaml.safe_load(f)
        pprint.pprint(config)

    trainer = Trainer(config,
                      outdir,
                      modeldir,
                      args.data_path,
                      args.sketchKeras_path,
                      args.iteration_count)
    trainer()
