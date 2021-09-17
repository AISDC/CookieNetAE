import torch

class CookieAE128ch(torch.nn.Module):
    def __init__(self, with_loss=False):
        super().__init__()
        self.with_loss = with_loss

        enc_ops = []
        enc_out_chs = (16, 32, 64, 128)
        enc_in_chs  = (1, ) + enc_out_chs[:-1]
        for _i, (ic, oc) in enumerate(zip(enc_in_chs, enc_out_chs)):
            enc_ops.append(torch.nn.Conv2d(ic, oc, kernel_size=3, padding=1))
            enc_ops.append(torch.nn.ReLU())
            if _i < len(enc_out_chs) - 1:
                enc_ops.append(torch.nn.MaxPool2d(2))
        self.encoder = torch.nn.Sequential(*enc_ops)
        
        dec_ops = []
        dec_out_chs = (128, 64, 32, 16)
        dec_in_chs  = (enc_out_chs[-1], ) + dec_out_chs[:-1]
        for _i, (ic, oc) in enumerate(zip(dec_in_chs, dec_out_chs)):
             # different with tf version which used asym padding
            if _i < len(enc_out_chs) - 1:
                dec_ops.append(torch.nn.ConvTranspose2d(ic, oc, kernel_size=2, stride=2))
            else:
                dec_ops.append(torch.nn.ConvTranspose2d(ic, oc, kernel_size=3, stride=1, padding=1))
            dec_ops.append(torch.nn.ReLU())
            
        dec_ops.append(torch.nn.Conv2d(dec_out_chs[-1], 1, kernel_size=3, padding=1))
        dec_ops.append(torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(*dec_ops)

        if self.with_loss:
            self.criterion = torch.nn.MSELoss()

    def forward(self, x, y=None):
        emb = self.encoder(x)
        rec = self.decoder(emb)

        if self.with_loss:
            mbsz = x.shape[0]
            # MSE SN implementation only supports 2d so far
            loss = self.criterion(rec.view(mbsz, -1), y.view(mbsz, -1))
            return loss, rec
        else:
            return rec

class CookieAE16ch(torch.nn.Module):
    def __init__(self, with_loss=False):
        super().__init__()
        self.with_loss = with_loss

        enc_ops = []
        enc_out_chs = (16, 32, 64, 128)
        enc_in_chs  = (1, ) + enc_out_chs[:-1]
        for _i, (ic, oc) in enumerate(zip(enc_in_chs, enc_out_chs)):
            enc_ops.append(torch.nn.Conv2d(ic, oc, kernel_size=3, padding=1))
            enc_ops.append(torch.nn.ReLU())
            if _i < len(enc_out_chs) - 1:
                enc_ops.append(torch.nn.MaxPool2d(2))
        self.encoder = torch.nn.Sequential(*enc_ops)
        
        dec_ops = []
        dec_out_chs = (128, 64, 32, 16)
        dec_in_chs  = (enc_out_chs[-1], ) + dec_out_chs[:-1]
         # different with tf version which used asym padding
        for _i, (ic, oc) in enumerate(zip(dec_in_chs, dec_out_chs)):
            if _i < len(enc_out_chs) - 1:
                dec_ops.append(torch.nn.ConvTranspose2d(ic, oc, kernel_size=(1, 2), stride=(1, 2)))
            else:
                dec_ops.append(torch.nn.ConvTranspose2d(ic, oc, kernel_size=3, stride=1, padding=1))
            dec_ops.append(torch.nn.ReLU())
            
        dec_ops.append(torch.nn.Conv2d(dec_out_chs[-1], 1, kernel_size=3, padding=1))
        dec_ops.append(torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(*dec_ops)
        
        if self.with_loss:
            self.criterion = torch.nn.MSELoss()
            
    def forward(self, x, y=None):
        emb = self.encoder(x)
        rec = self.decoder(emb)

        if self.with_loss:
            mbsz = x.shape[0]
            # MSE SN implementation only supports 2d so far
            loss = self.criterion(rec.view(mbsz, -1), y.view(mbsz, -1))
            return loss, rec
        else:
            return rec