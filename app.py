import sys, argparse, random, os, gc, requests, json, piexif, discord, torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from fold_to_ascii import fold
from discord.ext import commands
import functools, typing, asyncio
from requests.structures import CaseInsensitiveDict
from io import BytesIO

metadata = PngInfo()
def parse_args():
    parser = argparse.ArgumentParser(description="hf api")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="lilpotat/a3",
        help='model',
    )
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    return args
args = parse_args()

def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        wrapped = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, wrapped)
    return wrapper

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

root_folder = 'images'
model_folder = args.model

if os.path.exists(f"{root_folder}") == False:
    os.mkdir(f"{root_folder}")
image_folder = max([int(f) for f in os.listdir(f"{root_folder}")], default=0)
if os.path.exists(f"{root_folder}/{image_folder:04}") == False:
    os.mkdir(f"{root_folder}/{image_folder:04}")
name = max([int(f[: f.index(".")]) for f in os.listdir(f"{root_folder}/{image_folder:04}")], default=0)

pipe = StableDiffusionPipeline.from_pretrained(model_folder, torch_dtype=torch.float16, safety_checker=None, custom_pipeline="lpw_stable_diffusion").to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

@to_thread
def generate(discord_token, discord_channel_id, discord_user, by, num_inference_steps, guidance_scale, sampler, width, height, prompt, negative_prompt, suffix, image_folder, name):
    width = closestNumber(width, 8)
    height = closestNumber(height, 8)
    metadata.add_text("Prompt", f"{prompt}")
    metadata.add_text("by", f"{by}")
    gc.collect()
    real_seed = torch.cuda.initial_seed()
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, height=height, width=width, guidance_scale=guidance_scale).images[0]
    if(suffix == 'png'):
      image.save(f"{root_folder}/{image_folder:04}/{name:04}.{suffix}", pnginfo=metadata)
    else:
      zeroth_ifd = {piexif.ImageIFD.ImageDescription: f"{fold(prompt)}", piexif.ImageIFD.Make: f"{fold(by)}", piexif.ImageIFD.Model: f"{model_folder}", piexif.ImageIFD.Copyright: f"Attribution 4.0 International (CC BY 4.0)"}
      exif_dict = {"0th": zeroth_ifd}
      exif_bytes = piexif.dump(exif_dict)
      image.save(f"{root_folder}/{image_folder:04}/{name:04}.{suffix}", "JPEG", quality=70, exif=exif_bytes)
    files = {f"{image_folder:04}_{name:04}.{suffix}": open(f"{root_folder}/{image_folder:04}/{name:04}.{suffix}", "rb").read()}
    payload = {"content": f"{prompt}\nSteps: {num_inference_steps}, Sampler: {sampler}, CFG scale: {guidance_scale}, Seed: {real_seed}, Size: {width}x{height}, Model: {model_folder} - {discord_user}"}
    requests.post(f"https://discord.com/api/v9/channels/{discord_channel_id}/messages", data=payload, headers={"authorization": f"Bot {discord_token}"}, files=files)
    os.remove(f"{root_folder}/{image_folder:04}/{name:04}.{suffix}")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='>', intents=intents)

discord_token = os.environ["DISCORD_TOKEN"]

@bot.command()
async def L(ctx, *, message: str):
    await generate(discord_token, ctx.channel.id, ctx.message.author.mention, "camenduru", 50, 7.5, "EulerA", 512, 512, message, "nsfw", "png", image_folder, name)
bot.run(discord_token)
