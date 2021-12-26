from transformers import pipeline
import torch
print('start')
# classifier = pipeline('sentiment-analysis')
# classifier('We are very happy to show you the 🤗 Transformers library.')

# results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
# for result in results:
    # print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# from transformers import ViTFeatureExtractor, ViTForImageClassification
# from PIL import Image
# import requests

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# inputs = feature_extractor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])

def run():
    torch.multiprocessing.freeze_support()
    print('loop')
    classifier = pipeline('sentiment-analysis')
    results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
    for result in results:
        print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


if __name__ == '__main__':
    print('main')
    run()