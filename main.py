import os
import time
from pprint import pprint
import torch
from Cam import Camera
from PIL import Image, ImageTk, ExifTags
from torchvision import transforms, models
import matplotlib.pyplot as plt
from fpdf import FPDF
import concurrent.futures

camera = Camera()




# Blende Check
def compute_sharpness_apertures(camera, zoom_factor, num_captures=3):
    current_dir = os.getcwd()
    temp_dir = os.path.join(current_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        apertures = camera.listAperture()
        image_data_dict = {}
        sharpest_images = {}

        zoom_size = int(224 / zoom_factor)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])

        def compute_sharpness(image):
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')
            with torch.no_grad():
                features = model(input_batch)
            sharpness = torch.mean(features).item()
            return sharpness

        for i, aperture in enumerate(apertures):
            camera.setAperture(float(aperture))
            sharpest_img = None
            sharpest_value = None
            for _ in range(num_captures):
                try:
                    file_path = camera.capture()
                    with Image.open(file_path) as raw_image:
                        zoomed_img = raw_image.crop((
                            int(raw_image.width / 2) - zoom_size,
                            int(raw_image.height / 2) - zoom_size,
                            int(raw_image.width / 2) + zoom_size,
                            int(raw_image.height / 2) + zoom_size
                        ))
                        sharpness = compute_sharpness(zoomed_img) * 1000
                        if sharpest_value is None or sharpness > sharpest_value:
                            sharpest_value = sharpness
                            sharpest_img = zoomed_img
                        zoomed_img.save(os.path.join(temp_dir, f"zoomed_image_{aperture}_{_}.png"))
                    os.remove(file_path)
                except FileNotFoundError:
                    print(f"File not found for aperture {aperture} on capture attempt {_}. Skipping this capture.")
                    continue
            image_data_dict[float(aperture)] = sharpest_value
            sharpest_images[float(aperture)] = sharpest_img
            print(f"Aperture: {aperture}, Max Sharpness: {sharpest_value}")

        sharpest_aperture = max(image_data_dict, key=image_data_dict.get)
        print(f"Sharpest aperture: {sharpest_aperture}")

        # Check if the sharpest aperture is within the last few values
        aperture_values = list(image_data_dict.keys())
        last_aperture_range = aperture_values[-4:]  # Last 4 aperture values
        if sharpest_aperture in last_aperture_range:
            print(
                f"AF Fine Tuning may be recommended to achieve the best sharpness, as the sharpness is within the narrowest range of the aperture, specifically at ({sharpest_aperture}). This means that the focus settings need to be fine-tuned.")

        plt.figure()
        plt.plot(apertures, list(image_data_dict.values()))
        plt.title('Sharpness vs. Aperture')
        plt.xlabel('Aperture')
        plt.ylabel('Sharpness')
        plt.grid(True)
        plt.xticks(ticks=range(len(apertures)), labels=apertures, rotation=45)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.join(current_dir, 'sharpness_vs_aperture.png'), format='png')

        pdf = FPDF()
        for aperture, img in sharpest_images.items():
            img_path = os.path.join(temp_dir, f"sharpest_image_{aperture}.png")
            img.save(img_path)
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Aperture: {aperture}", ln=True, align='C')
            pdf.image(img_path, x=10, y=30, w=190)
        pdf.output("Sharpest_Images.pdf")
    finally:
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))

    return sharpest_aperture

# Iso Check
def compute_noise_iso(camera, zoom_factor, num_captures=3):
    current_dir = os.getcwd()
    temp_dir = os.path.join(current_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        iso_conversion = {
            "50": 50,
            "64": 64,
            "80": 80,
            "100": 100,
            "125": 125,
            "160": 160,
            "200": 200,
            "250": 250,
            "320": 320,
            "400": 400,
            "500": 500,
            "640": 640,
            "800": 800,
            "1000": 1000,
            "1250": 1250,
            "1600": 1600,
            "2000": 2000,
            "2500": 2500,
            "3200": 3200,
            "4000": 4000,
            "5000": 5000,
            "6400": 6400,
            "Hi 0.3": 8000,
            "Hi 0.7": 10000,
            "Hi 1": 14400,
            "Hi 2": 25600
        }

        iso_values = []
        for iso in camera.listIso():
            if iso.startswith("Hi"):
                iso_value = iso_conversion.get(iso, None)
                if iso_value is not None:
                    iso_values.append(iso_value)
            else:
                iso_values.append(int(iso))

        image_data_dict = {}
        noisiest_images = {}

        zoom_size = int(224 / zoom_factor)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model = models.vgg16(pretrained=True)
        model = torch.nn.Sequential(*list(model.features.children())[:23])

        def compute_noise(image):
            gray_image = image.convert("L")
            pixel_values = list(gray_image.getdata())
            differences = [abs(pixel_values[i] - pixel_values[i+1]) for i in range(len(pixel_values)-1)]
            noise = sum(differences) / len(differences)
            return noise

        for i, iso in enumerate(iso_values):
            camera.setIso(str(iso))
            noisiest_img = None
            noisiest_value = None
            for _ in range(num_captures):
                try:
                    file_path = camera.capture()
                    with Image.open(file_path) as raw_image:
                        zoomed_img = raw_image.crop((
                            int(raw_image.width / 2) - zoom_size,
                            int(raw_image.height / 2) - zoom_size,
                            int(raw_image.width / 2) + zoom_size,
                            int(raw_image.height / 2) + zoom_size
                        ))
                        noise = compute_noise(zoomed_img) * 1000
                        if noisiest_value is None or noise > noisiest_value:
                            noisiest_value = noise
                            noisiest_img = zoomed_img
                        zoomed_img.save(os.path.join(temp_dir, f"zoomed_image_iso_{iso}_{_}.png"))
                    os.remove(file_path)
                except FileNotFoundError:
                    print(f"File not found for ISO {iso} on capture attempt {_}. Skipping this capture.")
                    continue
            image_data_dict[iso] = noisiest_value
            noisiest_images[iso] = noisiest_img
            print(f"ISO: {iso}, Max Noise: {noisiest_value}")

        noisiest_iso = max(image_data_dict, key=image_data_dict.get)
        print(f"Noisiest ISO: {noisiest_iso}")

        iso_values = list(image_data_dict.keys())
        iso_values = [int(iso) if isinstance(iso, str) else iso for iso in iso_values]

        plt.figure(figsize=(10, 6))
        plt.plot(iso_values, list(image_data_dict.values()), marker='o', linestyle='-', color='blue')
        plt.title('Noise vs. ISO')
        plt.xlabel('ISO')
        plt.ylabel('Noise')
        plt.grid(True)
        plt.xticks(ticks=iso_values)
        plt.tight_layout()
        plt.savefig(os.path.join(temp_dir, 'noise_vs_iso.png'), format='png')

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Noise vs. ISO Chart", ln=True, align='C')
        pdf.image(os.path.join(temp_dir, 'noise_vs_iso.png'), x=10, y=20, w=190)
        pdf.add_page()

        image_width = 50
        image_height = 50
        spacing = 10
        x_start = 10
        y_start = 20

        for iso, img in noisiest_images.items():
            img_path = os.path.join(temp_dir, f"noisiest_image_iso_{iso}.png")
            img.save(img_path)
            pdf.image(img_path, x_start, y_start, image_width, image_height)
            pdf.set_font("Arial", size=10)
            pdf.cell(image_width, 10, txt=f"ISO: {iso}", ln=True, align='C')
            if x_start + image_width + spacing < pdf.w - image_width:
                x_start += image_width + spacing
            else:
                x_start = 10
                y_start += image_height + spacing

        pdf.output(os.path.join(current_dir, 'noisiest_images.pdf'), 'F')

    finally:
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))

    return noisiest_iso

# AF Fine Tuning ( Funktioniert nicht so gut )
def compute_sharpness_fine_tuning(camera, zoom_factor, num_captures=3):
    current_dir = os.getcwd()
    temp_dir = os.path.join(current_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        image_data_dict = {}
        sharpest_images = {}

        zoom_size = int(224 / zoom_factor)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model = models.vgg16(pretrained=True)
        model = torch.nn.Sequential(*list(model.features.children())[:23])

        def compute_sharpness(image):
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            with torch.no_grad():
                features = model(input_batch)
            sharpness = torch.mean(features).item()
            return sharpness

        def prompt_fine_tuning_direction(step):
            if step < 0:
                return f"Set the Focus Fine-tuning on the camera in the negative direction ({step}), then press Enter to continue."
            elif step > 0:
                return f"Set the Focus Fine-tuning on the camera in the positive direction ({step}), then press Enter to continue."
            else:
                return f"Set the Focus Fine-tuning on the camera to the starting point ({step}), then press Enter to continue."

        steps = [0, -10, -20, 10, 20]
        for step in steps:
            input(prompt_fine_tuning_direction(step))
            sharpest_img = None
            sharpest_value = None
            for _ in range(num_captures):
                try:
                    file_path = camera.capture()
                    with Image.open(file_path) as raw_image:
                        zoomed_img = raw_image.crop((
                            int(raw_image.width / 2) - zoom_size,
                            int(raw_image.height / 2) - zoom_size,
                            int(raw_image.width / 2) + zoom_size,
                            int(raw_image.height / 2) + zoom_size
                        ))
                        sharpness = compute_sharpness(zoomed_img) * 1000
                        if sharpest_value is None or sharpness > sharpest_value:
                            sharpest_value = sharpness
                            sharpest_img = zoomed_img
                        zoomed_img.save(os.path.join(temp_dir, f"zoomed_image_{step}_{_}.png"))
                    os.remove(file_path)
                except FileNotFoundError:
                    print(f"File not found for fine-tuning {step} on capture attempt {_}. Skipping this capture.")
                    continue
            image_data_dict[step] = sharpest_value
            sharpest_images[step] = sharpest_img
            print(f"Focus Fine-tuning: {step}, Max Sharpness: {sharpest_value}")

        sharpest_fine_tuning = max(image_data_dict, key=image_data_dict.get)
        print(f"Sharpest focus fine-tuning: {sharpest_fine_tuning}")

        plt.figure()
        plt.plot(steps, list(image_data_dict.values()))
        plt.title('Sharpness vs. Focus Fine-tuning')
        plt.xlabel('Focus Fine-tuning')
        plt.ylabel('Sharpness')
        plt.grid(True)
        plt.xticks(ticks=steps, labels=steps, rotation=45)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.join(current_dir, 'sharpness_vs_fine_tuning.png'), format='png')

        # Create PDF
        pdf = FPDF()
        for fine_tuning_value, img in sharpest_images.items():
            img_path = os.path.join(temp_dir, f"sharpest_image_{fine_tuning_value}.png")
            img.save(img_path)
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Focus Fine-tuning: {fine_tuning_value}", ln=True, align='C')
            pdf.image(img_path, x=10, y=30, w=190)
        pdf.output("Sharpest_Images.pdf")
    finally:
        # Delete files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))

    return sharpest_fine_tuning

# Exif data auslesen
def get_exif_data(camera):
    image_path = "captured_image"
    camera.capture(image_path)
    img = Image.open(image_path + ".jpg")

    exif_data = img._getexif()
    extracted_data = {}
    pprint(exif_data)
    if exif_data is not None:
        for tag, value in exif_data.items():
            tagname = ExifTags.TAGS.get(tag, tag)
            if tagname in ["Model", "Software", "FocalLength", "SubjectDistanceRange", "ISOSpeedRatings", "FNumber", "ExposureTime"]:
                extracted_data[tagname] = value

    img.close()
    time.sleep(1)
    os.remove(image_path + ".jpg")
    return extracted_data





#compute_sharpness_fine_tuning(camera, 2, 5)

#compute_noise_iso(camera, 2, 1)

#compute_sharpness_apertures(camera, 3, 3)






