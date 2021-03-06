from django.shortcuts import render, HttpResponse
import base64
from django.http import JsonResponse
from torch import nn
from PIL import Image
import torch
from torchvision import transforms
from io import BytesIO


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate=0.5):
        super(LPRNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),  # *** 11 ***
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),  # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + class_num, out_channels=class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        keep_features = []
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)  # ?????????????????????
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]   # ?????????????????????????????????
                keep_features.append(x)

            global_context = []
            for i, f in enumerate(keep_features):
                # ??????????????????????????????
                if i in [0, 1]:
                    f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
                if i in [2]:
                    f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)

                # ?????????????????????BN????????????????????????????????????????????????
                f_pow = torch.pow(f, 2)
                f_mean = torch.mean(f_pow)
                f = torch.div(f, f_mean)
                global_context.append(f)

        # ?????????????????????
        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits


labels = ["???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???",
          "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
          "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
          "Y", "Z", "-"]


def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []
    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)
    return tuple(input_lengths), tuple(target_lengths)


device = torch.device('cpu')
test_batch_size = 120
T_length = 18  # ??????8??????????????????????????????????????????????????????????????????
pretrained_weights_path = '/root/carPlate/app/LPRNet__epoch_21.pth'


def getPlate(pilimage):
    preprocess_transform = transforms.Compose([
        transforms.Resize((24, 94)),
        transforms.ToTensor(),
    ])

    net = LPRNet(class_num=len(labels)).to(device)
    net.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    # print("load pretrained model successful!")
    img = pilimage.convert('RGB')
    img = preprocess_transform(img).to(device)
    net.eval()
    with torch.no_grad():
        img = img.unsqueeze(0)
        prebs = net(img)
        preb_labels = []
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = preb.argmax(dim=0)
            no_repeat_blank_label = []
            pre_c = preb_label[0]
            if pre_c != len(labels) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(labels) - 1):
                    if c == len(labels) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        plate = ''
        for i, label in enumerate(preb_labels):
            for j in label:
                plate += labels[j]
        return plate


img_path = '2.png'


def carPlate(request):
    if request.method == "POST":
        if (request.POST.get('img')):
            img_data = base64.b64decode(request.POST.get('img'))
            pilimage = Image.open(BytesIO(img_data))
            plate = getPlate(pilimage)
            plate_json = {'plate': plate}
            return JsonResponse(plate_json)
        else:
            return JsonResponse({'plate':' '})
    return render(request, 'form.html')


