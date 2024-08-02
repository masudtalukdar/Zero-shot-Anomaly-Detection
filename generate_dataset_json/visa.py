import os
import json
import pandas as pd

class VisaSolver(object):
    CLSNAMES = [
        'pcb1', 'pcb2', 'pcb3',
        'pcb4'
    ]

    def __init__(self, root='./data/Visa/'):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test']
        self.csv_data = pd.read_csv(f'{root}/split_csv/1cls.csv', header=0)

    def run(self):
        columns = self.csv_data.columns  # [object, split, label, image, mask]
        info = {phase: {} for phase in self.phases}
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_data = self.csv_data[self.csv_data[columns[0]] == cls_name]
            for phase in self.phases:
                cls_info = []
                cls_data_phase = cls_data[cls_data[columns[1]] == phase]
                cls_data_phase.index = list(range(len(cls_data_phase)))
                for idx in range(cls_data_phase.shape[0]):
                    data = cls_data_phase.iloc[idx]  # Changed from data[idx]
                    is_abnormal = True if data.iloc[2] == 'anomaly' else False  # Changed from data[2]
                    info_img = dict(
                        img_path=data.iloc[3],  # Changed from data[3]
                        mask_path=data.iloc[4] if is_abnormal else '',  # Changed from data[4]
                        cls_name=cls_name,
                        specie_name='',
                        anomaly=1 if is_abnormal else 0,
                    )
                    cls_info.append(info_img)
                    if phase == 'test':
                        if is_abnormal:
                            anomaly_samples += 1
                        else:
                            normal_samples += 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)

if __name__ == '__main__':
    runner = VisaSolver(root='./data/Visa/')
    runner.run()
