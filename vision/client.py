import doctr
import cv2
import json 
import os, pathlib
import argparse

image_ext = ("*.jpg", "*.jpeg", "*.png")



from pdf2image import convert_from_path
import os
import glob
def pdf_to_images_renamed(pdf_path, image_extension="jpg",output_dir='../source'):
    if not isinstance(pdf_path,list):
    
    # Extract the PDF name without extension
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        if not output_dir:
            output_dir=os.path.split(pdf_path)[0]
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)
        max_image_n=sorted(glob.glob(output_dir+'/image_*'),key=lambda x: int(x.split('.')[0].split('_')[-1]))
        max_image_n=max_image_n[-1].split('.')[0].split('_')[-1] if max_image_n else '0'
        # Save each image with the format <pdf_name>_<page_no>.<image_extension>
        for page_no, image in enumerate(images, start=1):
            max_image_n=str(int(max_image_n)+1)
            image_filename = f"image_{'0'*(3-len(max_image_n)) if 3-len(max_image_n)>0 else ''}{max_image_n}.{image_extension}"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path, image_extension.upper())  # Save the image in the desired format
            print(f"Saved: {image_path}")
        
        print(f"All pages from {pdf_path} are converted to images in {output_dir}.")
    else:
        [pdf_to_images(path,image_extension=image_extension,output_dir=output_dir) for path in pdf_path]



def pdf_to_images(pdf_path, image_extension="z"):
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

        res_all=[]
        if isinstance(image, str):
            # if image.endswith('pdf'):
            #     image = doctr.io.DocumentFile(image)
            # else:
            #     image = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)]
            image = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)]
            H,W,C=image[0].shape    
            results = [self.model(image)]

        elif isinstance(image, list):
            if isinstance(image[0], str):
                image=[cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB) for im in image]
            results=self.model(image)
        for result,im in zip(results,image):
            H,W,C=im.shape    
            id_counter = 0
            res=[]
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:  # The bounding box geometry of the word
                            text_value = word.value  # The recognized text value
                            (x_min, y_min), (x_max, y_max) = word.geometry  # Convert geometry to bounding box coordinates

                            # Create the result dictionary in the desired format
                            result_dict = {
                                "id": id_counter,
                                "text": text_value,
                                "label": "",  # Assuming label is static here, update if needed
                                "box": [x_min, y_min, x_max, y_max],
                                "linking": []  # Example linking logic; modify as needed
                            }
                            res.append(result_dict)
                            id_counter += 1  # Increment ID counter for each word
            res_all.append(res)
        return res


def find_images(folder,output_dir=None):
    import glob
    pdf_to_images_renamed(glob.glob(os.path.join(folder,'*.pdf')),image_extension='png',output_dir=output_dir)
    print(pathlib.Path(os.path.realpath(output_dir)).rglob('*.png'))
    return [
        
        str(path)
        for ext in image_ext
        for path in pathlib.Path(os.path.realpath(output_dir)).rglob(ext) if not os.path.exists(str(path).replace(ext[2:],'json')) 
    ]



def main():

    parser = argparse.ArgumentParser(
        description='Detect Text on Images using vision. Results will be written as "<image>.json"'
    )
    parser.add_argument("--image_directory","-i", type=str, help="path to the directory containing the images",default='../source')
    parser.add_argument("--output_dir","-o", type=str, help="where to save images and json",default='../source')
    
    args = parser.parse_args()

    # get the image path
    all_images = find_images(args.image_directory,args.output_dir)
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
