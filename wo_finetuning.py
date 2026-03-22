from diffusers import StableDiffusionPipeline
import torch
import os

SUBJECT_NAMES = (
    "backpack", "backpack_dog", "bear_plushie", "berry_bowl", "can",
    "candle", "cat", "cat", "clock", "colorful_sneaker",
    # "dog", "dog", "dog", "dog", "dog",
    # "dog", "dog", "duck_toy", "fancy_boot", "grey_sloth_plushie",
    # "monster_toy", "pink_sunglasses", "poop_emoji", "rc_car", "red_cartoon",
    # "robot_toy", "shiny_sneaker", "teapot", "vase", "wolf_plushie",
)

# prompt_test_list = (
#     "a ${subject_name} in the jungle",
#     "a ${subject_name} in the snow",
#     "a ${subject_name} on the beach",
#     "a ${subject_name} on a cobblestone street",
#     "a ${subject_name} on top of pink fabric",
#     "a ${subject_name} on top of a wooden floor",
#     "a ${subject_name} with a city in the background",
#     "a ${subject_name} with a mountain in the background",
#     "a ${subject_name} with a blue house in the background",
#     "a ${subject_name} on top of a purple rug in a forest",
#     "a ${subject_name} wearing a red hat",
#     "a ${subject_name} wearing a santa hat",
#     "a ${subject_name} wearing a rainbow scarf",
#     "a ${subject_name} wearing a black top hat and a monocle",
#     "a ${subject_name} in a chef outfit",
#     "a ${subject_name} in a firefighter outfit",
#     "a ${subject_name} in a police outfit",
#     "a ${subject_name} wearing pink glasses",
#     "a ${subject_name} wearing a yellow shirt",
#     "a ${subject_name} in a purple wizard outfit",
#     "a red ${subject_name}",
#     "a purple ${subject_name}",
#     "a shiny ${subject_name}",
#     "a wet ${subject_name}",
#     "a cube shaped ${subject_name}",
# )

prompt_test_list=(
    # "a ${subject_name} in the jungle",
    # "a ${subject_name} in the snow",
    # "a ${subject_name} on the beach",
    # "a ${subject_name} on a cobblestone street",
    # "a ${subject_name} on top of pink fabric",
    # "a ${subject_name} on top of a wooden floor",
    # "a ${subject_name} with a city in the background",
    # "a ${subject_name} with a mountain in the background",
    # "a ${subject_name} with a blue house in the background",
    # "a ${subject_name} on top of a purple rug in a forest",
    "a ${subject_name} with a wheat field in the background",
    "a ${subject_name} with a tree and autumn leaves in the background",
    "a ${subject_name} with the Eiffel Tower in the background",
    "a ${subject_name} floating on top of water",
    "a ${subject_name} floating in an ocean of milk",
    "a ${subject_name} on top of green grass with sunflowers around it",
    "a ${subject_name} on top of a mirror",
    "a ${subject_name} on top of the sidewalk in a crowded street",
    "a ${subject_name} on top of a dirt road",
    "a ${subject_name} on top of a white rug",
    # "a red ${subject_name}",
    # "a purple ${subject_name}",
    # "a shiny ${subject_name}",
    # "a wet ${subject_name}",
    # "a cube shaped ${subject_name}",
  )

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

for subject in SUBJECT_NAMES:
    for scene in prompt_test_list:
        prompt = scene.replace("${subject_name}", subject)
        image = pipe(prompt).images[0]

        # 清理文件名中的非法字符
        safe_prompt = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")
        output_dir = f"./data/output/raw/{subject}_boft_raw"
        os.makedirs(output_dir, exist_ok=True)
        image.save(f"{output_dir}/{safe_prompt}.png")
