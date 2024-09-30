import doctr
import cv2
import json 
import os, pathlib
import argparse

image_ext = ("*.jpg", "*.jpeg", "*.png")



from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, image_extension="png"):
    if not isinstance(pdf_path,list):
    
    # Extract the PDF name without extension
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir=os.path.split(pdf_path)[0]
        
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)
        
        # Save each image with the format <pdf_name>_<page_no>.<image_extension>
        for page_no, image in enumerate(images, start=1):
            image_filename = f"{pdf_name}_{page_no}.{image_extension}"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path, image_extension.upper())  # Save the image in the desired format
            print(f"Saved: {image_path}")
        
        print(f"All pages from {pdf_path} are converted to images in {output_dir}.")
    else:
        [pdf_to_images(path) for path in pdf_path]





class VisionClient:
    def __init__(self):
        
        self.model=doctr.models.ocr_predictor('fast_base','parseq',pretrained=True)

    

    def infer(self, image):

        res = []
        if isinstance(image, str):
            # if image.endswith('pdf'):
            #     image = doctr.io.DocumentFile(image)
            # else:
            #     image = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)]
            image = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)]
        H,W,C=image[0].shape    
        results = self.model(image)
        id_counter = 0
        for page in results.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:  # The bounding box geometry of the word
                        text_value = word.value  # The recognized text value
                        (x_min, y_min), (x_max, y_max) = word.geometry  # Convert geometry to bounding box coordinates

                        # Create the result dictionary in the desired format
                        result_dict = {
                            "id": id_counter,
                            "text": text_value,
                            "label": "label",  # Assuming label is static here, update if needed
                            "box": [x_min*W, y_min*H, x_max*W, y_max*H],
                            "linking": []  # Example linking logic; modify as needed
                        }
                        res.append(result_dict)
                        id_counter += 1  # Increment ID counter for each word

        return res


def find_images(folder):
    import glob
    pdf_to_images(glob.glob(os.path.join(folder,'*.pdf')))
    return [
        
        str(path)
        for ext in image_ext
        for path in pathlib.Path(os.path.realpath(folder)).rglob(ext) if not os.path.exists(path.replace(ext[2:],json)) 
    ]


def main():

    parser = argparse.ArgumentParser(
        description='Detect Text on Images using vision. Results will be written as "<image>.json"'
    )
    parser.add_argument("--image_directory","-i", type=str, help="path to the directory containing the images",default='../source')
    args = parser.parse_args()

    # get the image path
    all_images = find_images(args.image_directory)
    # init the client
    vision_client = VisionClient()

    # do all the requests
    for image_path in all_images:
        img_name, _ = os.path.splitext(image_path)
        resp_name = "{}.json".format(img_name)
        result_dict=vision_client.infer(image_path)
        resp_js=json.dumps(result_dict)
        with open(resp_name, "w") as response:
            response.write(resp_js)


if __name__ == "__main__":
    main()
