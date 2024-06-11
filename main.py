import os
import openai
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import textwrap
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# TEXT COMPLETION
def gen_text(api_key, model_name, text_prompt, max_tokens, temperature, debug=False):
    openai.api_key = api_key
    completion = openai.ChatCompletion.create(model=model_name,
                                              messages=[{"role": "user", "content": text_prompt}],
                                              max_tokens=max_tokens, temperature=temperature)
    quote = completion.choices[0].message["content"].split("\n")
    clean_quotes = list(filter(None, quote))

    if debug:
        for q in clean_quotes:
            print(q)

    return clean_quotes


def gen_imgs(img_prompt, num_imgs, img_size):
    # IMAGE GENERATION
    openai.api_key = API_KEY
    list_o_imgs = []
    response = openai.Image.create(prompt=img_prompt, n=num_imgs, size=img_size)

    for n_img in range(num_imgs):
        img = url_to_img(response["data"][n_img]["url"])
        list_o_imgs.append(img)

    return list_o_imgs


def url_to_img(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def variations_of_existing_img(img, img_name, num_variations, img_size):
    img.save(img_name)
    response = openai.Image.create_variation(image=open(img_name, "rb"),
                                             n=num_variations,
                                             size=img_size)

    image_url = url_to_img(response['data'][0]['url'])
    print(response['data'][0]['url'])
    plt.imshow(image_url)
    return image_url


def find_dominant_contrast_colors(img):
    #Resizing parameters for faster processing
    width, height = 250, 250
    image = img
    image = image.resize((width, height), resample = 0)
    #Get colors from image object
    pixels = image.getcolors(width * height)
    #Sort them by count number(first element of tuple)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])
    #Get the most frequent color
    dominant_color = sorted_pixels[-1][1]

    contrast_color_scheme = simple_contrast(dominant_color)

    return dominant_color, contrast_color_scheme


def simple_contrast(contrast_color_scheme):
    r = 255 - contrast_color_scheme[0]
    g = 255 - contrast_color_scheme[1]
    b = 255 - contrast_color_scheme[2]
    return (r, g, b)


def text_on_img(list_o_images, list_o_quotes):
    myFont = ImageFont.truetype('JMH Typewriter-Bold.ttf', 39)

    for i_idx, img in enumerate(list_o_images):
        dominant_color, contrast_color = find_dominant_contrast_colors(img)

        for q_idx, quote in enumerate(list_o_quotes):
            img_name = "unInspired_imgs/" + "img" + str(i_idx) + "_quote" + str(q_idx) + ".jpg"
            copied_img = img.copy()

            # Call draw Method to add 2D graphics in an image
            I1 = ImageDraw.Draw(copied_img)
            print("quote: ", quote)
            if len(quote) > 10:
                quote = textwrap.fill(text=quote.split(' "')[1].replace('"', ''), width=40)
                random_number = random.randint(50, 800)
                (x, y) = (50, random_number)

                # Draw outline
                for adj in range(-3, 4):
                    # We create a shadow for each x, y offset
                    I1.text((x + adj, y), quote, font=myFont, fill=dominant_color)
                    I1.text((x, y + adj), quote, font=myFont, fill=dominant_color)

                # Add Text to an image
                I1.text((x, y), quote, font=myFont, fill=contrast_color)

                copied_img.save(img_name)


places = ["a scenic vista", "a serene lake", "a peaceful meadow", "a tranquil forest", "a quiet beach", "a rushing river",
           "a beautiful garden", "a majestic mountain", "a breathtaking waterfall", "a peaceful valley", "a tranquil pond",
           "a serene island", "a quiet desert", "a beautiful sunrise", "a majestic sunset", "a breathtaking sky", "a peaceful ocean",
           "a tranquil stream", "a quiet field", "a beautiful flower", "a majestic tree", "a breathtaking landscape",
           "a peaceful park", "a tranquil path", "a quiet road", "a beautiful city", "a majestic town", "a breathtaking village",
           "a peaceful country", "a tranquil world", "a quiet universe", "a beautiful planet", "a majestic galaxy", "a breathtaking cosmos",
           "a peaceful dimension", "a tranquil realm", "a quiet space", "a beautiful place", "a majestic location", "a breathtaking spot",
            "a peaceful setting", "a tranquil scene", "a quiet view", "a beautiful panorama", "a majestic vista", "a breathtaking landscape",
            "a peaceful view", "a tranquil scene", "a quiet panorama", "a beautiful vista", "a majestic landscape", "a breathtaking view",
            "a peaceful panorama", "a tranquil vista", "a quiet landscape", "a beautiful scene", "a majestic panorama", "a breathtaking vista",
            "a peaceful landscape", "a tranquil panorama", "a quiet vista", "a beautiful landscape", "a majestic scene", "a breathtaking panorama",
            "a peaceful vista", "a tranquil landscape", "a quiet scene", "a beautiful panorama", "a majestic vista", "a breathtaking landscape",
            "a peaceful view", "a tranquil scene", "a quiet panorama", "a beautiful vista", "a majestic landscape", "a breathtaking view", "a peaceful panorama",
            "a tranquil vista", "a quiet landscape", "a beautiful scene", "a majestic panorama", "a breathtaking vista", "a peaceful landscape",
            "a tranquil panorama", "a quiet vista", "a beautiful landscape", "a majestic scene", "a breathtaking panorama", "a peaceful vista",
            "a tranquil landscape", "a quiet scene", "a beautiful panorama", "a majestic vista", "a breathtaking landscape", "a peaceful view",
            "a tranquil scene", "a quiet panorama", "a beautiful vista", "a majestic landscape", "a breathtaking view"]

features = ["colorfull clouds", "vibrant mountains", "bright sun", "shiny stars", "colorful flowers", "vibrant trees", "bright moon", "shiny water",
            "colorful sky", "vibrant ocean", "bright city", "shiny lights", "colorful buildings", "vibrant cars", "bright people", "shiny animals",
            "vibrant waterfall", "bright river", "shiny lake", "colorful forest", "vibrant desert", "bright beach", "shiny meadow", "colorful garden",]
styles = ["digital art", "oil painting", "watercolor painting", "abstract art", "impressionist art", "realistic art", "surreal art", "maximalist art",
          "modern art", "contemporary art", "cubist art", "futurist art", "expressionist art", "pop art", "post-impressionist art",]
dark_words = ["uninspirational", "dark", "gloomy", "sad", "dismal", "bleak", "dreary", "melancholy", "somber", "sorrowful", "mournful"]
darker_words = ["unmotivational", "despairing", "hopeless", "discouraging", "disheartening", "dispiriting", "dismaying", "demoralizing"]

place = random.choice(places)
feature = random.choice(features)
style = random.choice(styles)
dark_word = random.choice(dark_words)  
darker_word = random.choice(darker_words)

# env vars
API_KEY = os.environ["OPEN_AI_API_KEY"]
print("API_KEY", API_KEY)
TEXT_MODEL_NAME = "gpt-3.5-turbo"   # "text-davinci-003"
TEMP = 0.7  # temp 0-1, higher temp generates stranger text
MAX_TOKENS = 75  # response lenght
TEXT_PROMPT = f"Give me 10 {dark_word} and {darker_word} quotes"  # "Give me 10 inspirational and positive quotes"
IMG_PROMPT = f"{place}, {feature}, {style}"
num_imgs = 2
IMG_SIZE = "1024x1024"
black_color_scheme = (0, 0, 0)
white_color_scheme = (255, 255, 255)

quotes = gen_text(API_KEY, TEXT_MODEL_NAME, TEXT_PROMPT, MAX_TOKENS, TEMP)
images = gen_imgs(IMG_PROMPT, num_imgs, IMG_SIZE)

text_on_img(images, quotes)

print("Text Prompt: ", TEXT_PROMPT)
print("Image Prompt: ", IMG_PROMPT)
