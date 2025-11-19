import os
import json
from .qacap_dataset import create_caption_dataloader
from .qacap_model import QACaptioner


if __name__ == '__main__':
    # This is an example of how to use the QACaptioner
    # You would need to set up your configuration object (__C) and paths correctly.
    
    # Mock configuration for demonstration
    class MockC:
        def __init__(self):
            self.TASK = 'example_task'
            # IMPORTANT: Update these paths to your actual data locations
            self.QUESTION_PATH = {
                'train': '/ocean/projects/cis250215p/shared/fca/third_party/prophet/datasets/coco2014/annotations/train_questions.json'
            }
            self.IMAGE_DIR = {
                'train': '/ocean/projects/cis250215p/shared/fca/third_party/prophet/datasets/coco2014/train2014'
            }
            # This should contain the splits you want to process
            self.FEATURE_SPLIT = ['train'] 
            self.SUBSET_COUNT = 10

    __C = MockC()

    # Check if paths are valid before proceeding
    train_questions_exist = os.path.exists(__C.QUESTION_PATH['train'])
    train_images_exist = os.path.exists(__C.IMAGE_DIR['train'])

    if not (train_questions_exist and train_images_exist):
        print("="*50)
        print("WARNING: Update the placeholder paths in the `if __name__ == '__main__':` block")
        print("in 'src/qacap/captioner.py' to your actual data locations.")
        print(f"Current question path: {__C.QUESTION_PATH['train']} (Exists: {train_questions_exist})")
        print(f"Current image dir: {__C.IMAGE_DIR['train']} (Exists: {train_images_exist})")
        print("="*50)
    else:
        # Create the dataloader
        # We pass a transform to resize images, but the model's processor will handle the rest
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])

        dataloader = create_caption_dataloader(
            __C,
            split_name_list=['train'],
            batch_size=4, # Small batch size for demonstration
            transform=transform
        )

        # Initialize the captioner
        captioner = QACaptioner()

        # Generate captions
        generated_captions = captioner.generate_from_loader(dataloader)

        # Print the results
        print("\n--- Generated Captions ---")
        print(json.dumps(generated_captions, indent=4))

        # To save to a file:
        output_path = 'generated_captions.json'
        with open(output_path, 'w') as f:
            json.dump(generated_captions, f, indent=4)
        print(f"\nResults saved to {output_path}")