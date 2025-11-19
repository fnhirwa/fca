import os
import json
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class CommonData:
    """
    Loads data common to all splits, but for raw images instead of features.
    """
    def __init__(self, __C):
        print('Loading common data for raw images...')
        self.img_path_list = []
        # In prophet, FEATURE_SPLIT is used for splits containing image features.
        # Here we adapt it for directories containing raw images.
        for split in __C.FEATURE_SPLIT:
            # Assumes __C.IMAGE_DIR holds the paths to raw image directories
            if split in __C.IMAGE_DIR:
                # Correctly join the workspace path with the relative image directory
                image_dir = os.path.join(__C.PROPHET_PATH, __C.IMAGE_DIR[split])
                self.img_path_list.extend(glob.glob(os.path.join(image_dir, '*.jpg')))

        self.imgid_to_path = {}
        for path in self.img_path_list:
            try:
                # Assumes standard COCO/VQA naming like '..._..._000000123456.jpg'
                img_id = int(os.path.basename(path).split('_')[-1].split('.')[0])
                self.imgid_to_path[img_id] = path
            except (ValueError, IndexError):
                continue
        
        print(f'== Found {len(self.imgid_to_path)} images.')
        print('Common data process is done.\n')

class DataSet(Dataset):
    """
    A PyTorch Dataset for loading questions and corresponding raw images,
    mirroring the structure of the prophet's original DataSet.
    """
    def __init__(self, __C, common_data, split_name_list):
        self.__C = __C
        print(f'Loading dataset for {self.__C.TASK}|captioning({split_name_list})')

        self.imgid_to_path = common_data.imgid_to_path

        # Load questions
        self.questions = []
        for split_name in split_name_list:
            ques_path = os.path.join(__C.PROPHET_PATH, __C.QUESTION_PATH[split_name])
            # print(f"--- Loading questions from: {ques_path} ---")
            ques_data = json.load(open(ques_path, 'r'))
            if 'questions' in ques_data:
                self.questions.extend(ques_data['questions'])
            else:
                self.questions.extend(ques_data)
        
        # Filter questions to only include those with available images
        self.questions = [
            q for q in self.questions if q.get('image_id') in self.imgid_to_path
        ]

        # Optional per-split subsetting for quick testing
        subset_ratio = getattr(__C, 'SUBSET_RATIO', None)
        subset_count = getattr(__C, 'SUBSET_COUNT', None)

        if subset_count is not None:
            self.questions = self.questions[:subset_count]
        elif subset_ratio is not None:
            if subset_ratio <= 0 or subset_ratio > 1:
                raise ValueError('SUBSET_RATIO must be in (0, 1].')
            num_samples = max(1, int(len(self.questions) * subset_ratio))
            self.questions = self.questions[:num_samples]
        
        self.data_size = len(self.questions)
        print(f'== data size: {self.data_size}\n')

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        question_info = self.questions[idx]
        question_text = question_info['question']
        image_id = question_info['image_id']
        question_id = question_info['question_id']

        image_path = self.imgid_to_path.get(image_id)
        
        if not image_path or not os.path.exists(image_path):
            # This should ideally not happen due to pre-filtering
            return None


        # print(f"\n--- Loading item {idx} ---")
        # print(f"  - Question ID: {question_id}")
        # print(f"  - Question: '{question_text}'")
        # print(f"  - Image Path: {image_path}")
    

        # Return the raw PIL image. The processor will handle transforms.
        image = Image.open(image_path).convert('RGB')

        return {
            'image': image,
            'question': question_text,
            'image_id': image_id,
            'question_id': question_id,
            'image_path': image_path
        }

def caption_collate_fn(batch):
    """
    Custom collate function to handle batches of PIL images.
    The default collate function cannot stack PIL images into a tensor.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Separate images from other data
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    image_ids = torch.tensor([item['image_id'] for item in batch])
    question_ids = torch.tensor([item['question_id'] for item in batch])
    image_paths = [item['image_path'] for item in batch]

    return {
        'image': images, # List of PIL images
        'question': questions,
        'image_id': image_ids,
        'question_id': question_ids,
        'image_path': image_paths
    }

def create_caption_dataloader(__C, split_name_list, batch_size=8, shuffle=False, num_workers=0):
    """
    Creates a DataLoader for the question-image captioning task.
    No transform is applied here; the model's processor will handle it.
    """
    common_data = CommonData(__C)
    
    dataset = DataSet(
        __C,
        common_data,
        split_name_list,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=caption_collate_fn # Use the custom collate function
    )
    
    return dataloader
