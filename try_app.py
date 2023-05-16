from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import gradio as gr
import torch
from PIL import Image

model_id = 'SG161222/Realistic_Vision_V1.4'
prefix = 'RAW photo,'
     
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
  model_id,
  torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
  scheduler=scheduler)

pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
  model_id,
  torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
  scheduler=scheduler)

if torch.cuda.is_available():
  pipe = pipe.to("cuda")
  pipe_i2i = pipe_i2i.to("cuda")

def error_str(error, title="Error"):
    return f"""#### {title}
            {error}"""  if error else ""


def _parse_args(prompt, generator):
        parser = argparse.ArgumentParser(
            description="making it work."
        )
        parser.add_argument(
            "--no-half-vae", help="no half vae"
        )

        cmdline_args = parser.parse_args()
        command = cmdline_args.command
        conf_file = cmdline_args.conf_file
        conf_args = Arguments(conf_file)
        opt = conf_args.readArguments()

        if cmdline_args.config_overrides:
            for config_override in cmdline_args.config_overrides.split(";"):
                config_override = config_override.strip()
                if config_override:
                    var_val = config_override.split("=")
                    assert (
                        len(var_val) == 2
                    ), f"Config override '{var_val}' does not have the form 'VAR=val'"
                    conf_args.add_opt(opt, var_val[0], var_val[1], force_override=True)

def inference(prompt, guidance, steps, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt="", auto_prefix=False):
  generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None
  prompt = f"{prefix} {prompt}" if auto_prefix else prompt

  try:
    if img is not None:
      return img_to_img(prompt, neg_prompt, img, strength, guidance, steps, width, height, generator), None
    else:
      return txt_to_img(prompt, neg_prompt, guidance, steps, width, height, generator), None
  except Exception as e:
    return None, error_str(e)
      
      

def txt_to_img(prompt, neg_prompt, guidance, steps, width, height, generator):

    result = pipe(
      prompt,
      negative_prompt = neg_prompt,
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator)
    
    return result.images[0]

def img_to_img(prompt, neg_prompt, img, strength, guidance, steps, width, height, generator):

    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipe_i2i(
        prompt,
        negative_prompt = neg_prompt,
        init_image = img,
        num_inference_steps = int(steps),
        strength = strength,
        guidance_scale = guidance,
        width = width,
        height = height,
        generator = generator)
        
    return result.images[0]

    def fake_safety_checker(images, **kwargs):
      return result.images[0], [False] * len(images)
    
    pipe.safety_checker = fake_safety_checker

with gr.Blocks() as demo:
    gr.Markdown("Using: " + model_id)
    gr.Markdown("Using : GPU" if torch.cuda.is_available() else "Using : CPU")
    with gr.Row():
        
        with gr.Column(scale=55):
          with gr.Group():
              with gr.Row():
                prompt = gr.Textbox(label="Prompt", show_label=False,max_lines=2,placeholder=f"{prefix} [your prompt]")
                generate = gr.Button(value="Generate")

              image_out = gr.Image(height=512)
          error_output = gr.Markdown()

        with gr.Column(scale=45):
          with gr.Tab("Options"):
            with gr.Group():
              neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")
              auto_prefix = gr.Checkbox(label="Prefix styling tokens automatically (RAW photo,)", value=prefix, visible=prefix)

              with gr.Row():
                guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
                steps = gr.Slider(label="Steps", value=25, minimum=2, maximum=75, step=1)

              with gr.Row():
                width = gr.Slider(label="Width", value=512, minimum=64, maximum=1024, step=8)
                height = gr.Slider(label="Height", value=512, minimum=64, maximum=1024, step=8)

              seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)

          with gr.Tab("Image to image"):
              with gr.Group():
                image = gr.Image(label="Image", height=256, tool="editor", type="pil")
                strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)
    
    auto_prefix.change(lambda x: gr.update(placeholder=f"{prefix} [your prompt]" if x else "[Your prompt]"), inputs=auto_prefix, outputs=prompt, queue=False)

    inputs = [prompt, guidance, steps, width, height, seed, image, strength, neg_prompt, auto_prefix]
    outputs = [image_out, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

    

demo.queue(concurrency_count=1)
demo.launch()
