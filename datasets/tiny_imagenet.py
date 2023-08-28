import torchvision
import numpy as np
import os

class IMBALANETINYIMGNET(torchvision.datasets.ImageFolder):
    cls_num = 200

    def __init__(self, root='data/tiny-imagenet-200', imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None):
        split = 'train'
        if not train:
            split = 'val'
        data_dir = os.path.join(root, split)
        super(IMBALANETINYIMGNET, self).__init__(data_dir, transform=transform, target_transform=target_transform)
        np.random.seed(rand_number)

        if train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.class_freq = img_num_list
            self.gen_imbalanced_data(img_num_list)
        else:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, 1)
            self.class_freq = img_num_list
            self.gen_imbalanced_data(img_num_list)
            
        self.labels = self.targets
        print("{} Mode: Contain {} images".format(split, len(self.samples)))

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.samples) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            res_list = [self.samples[i] for i in selec_idx]
            new_data.extend(res_list)
            new_targets.extend([the_class, ] * the_img_num)
        self.samples = new_data
        self.targets = new_targets
    
    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
if __name__ == '__main__':
    tiny_imagenet_train = IMBALANETINYIMGNET(train=True)
    tiny_imagenet_val = IMBALANETINYIMGNET(train=False)
    
    
