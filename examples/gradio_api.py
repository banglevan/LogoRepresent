from gradio_client import Client

client = Client("http://127.0.0.1:7865/", output_dir='api_output', serialize=False)
result = client.predict(
				True,	# bool in 'Generate Image Grid for Each Batch' Checkbox component
				"Howdy!",	# str in 'parameter_11' Textbox component
				"Howdy!",	# str in 'Negative Prompt' Textbox component
				["Fooocus V2"],	# List[str] in 'Selected Styles' Checkboxgroup component
				"Quality",	# str in 'Performance' Radio component
				"704×1408 <span style='color: grey;'> ∣ 1:2</span>",	# str in 'Aspect Ratios' Radio component
				1,	# int | float (numeric value between 1 and 32) in 'Image Number' Slider component
				"png",	# str in 'Output Format' Radio component
				-1,	# str in 'Seed' Textbox component
				True,	# bool in 'Read wildcards in order' Checkbox component
				0,	# int | float (numeric value between 0.0 and 30.0) in 'Image Sharpness' Slider component
				1,	# int | float (numeric value between 1.0 and 30.0) in 'Guidance Scale' Slider component
				# str (Option from: ['animaPencilXL_v310.safetensors',
									# 'realisticStockPhoto_v20.safetensors',
									# 'sd_xl_refiner_1.0_0.9vae.safetensors'])
									# in 'Base Model (SDXL only)' Dropdown component
				"animaPencilXL_v310.safetensors",
				# str (Option from: ['None', 'animaPencilXL_v310.safetensors',
									# 'realisticStockPhoto_v20.safetensors',
									# 'sd_xl_refiner_1.0_0.9vae.safetensors'])
									# in 'Refiner (SDXL or SD 1.5)' Dropdown component
				"None",
				0.1,	# int | float (numeric value between 0.1 and 1.0) in 'Refiner Switch At' Slider component
				True,	# bool in 'Enable' Checkbox component
				"None",	# str (Option from: ['None', 'sd_xl_offset_example-lora_1.0.safetensors',
						# 'SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 'sdxl_hyper_sd_4step_lora.safetensors',
						# 'sdxl_lcm_lora.safetensors', 'sdxl_lightning_4step_lora.safetensors'])
						# in 'LoRA 1' Dropdown component
				-2,	# int | float (numeric value between -2 and 2)
					# in 'Weight' Slider component
				True,	# bool in 'Enable' Checkbox component
				"None",	# str (Option from: ['None', 'sd_xl_offset_example-lora_1.0.safetensors', 'SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 'sdxl_hyper_sd_4step_lora.safetensors', 'sdxl_lcm_lora.safetensors', 'sdxl_lightning_4step_lora.safetensors'])
						# in 'LoRA 2' Dropdown component
				-2,	# int | float (numeric value between -2 and 2)
					# in 'Weight' Slider component
				True,	# bool in 'Enable' Checkbox component
				"None",	# str (Option from: ['None', 'sd_xl_offset_example-lora_1.0.safetensors', 'SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 'sdxl_hyper_sd_4step_lora.safetensors', 'sdxl_lcm_lora.safetensors', 'sdxl_lightning_4step_lora.safetensors'])
						# in 'LoRA 3' Dropdown component
				-2,	# int | float (numeric value between -2 and 2)
					# in 'Weight' Slider component
				True,	# bool in 'Enable' Checkbox component
				"None",	# str (Option from: ['None', 'sd_xl_offset_example-lora_1.0.safetensors', 'SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 'sdxl_hyper_sd_4step_lora.safetensors', 'sdxl_lcm_lora.safetensors', 'sdxl_lightning_4step_lora.safetensors'])
						# in 'LoRA 4' Dropdown component
				-2,	# int | float (numeric value between -2 and 2)
					# in 'Weight' Slider component
				True,	# bool in 'Enable' Checkbox component
				"None",	# str (Option from: ['None', 'sd_xl_offset_example-lora_1.0.safetensors', 'SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 'sdxl_hyper_sd_4step_lora.safetensors', 'sdxl_lcm_lora.safetensors', 'sdxl_lightning_4step_lora.safetensors'])
						# in 'LoRA 5' Dropdown component
				-2,	# int | float (numeric value between -2 and 2)
					# in 'Weight' Slider component
				True,	# bool in 'Input Image' Checkbox component
				"Howdy!",	# str in 'parameter_94' Textbox component
				"Disabled",	# str in 'Upscale or Variation:' Radio component
				"",
								# str (filepath or URL to image)
								# in 'Image' Image component
				["Left"],	# List[str] in 'Outpaint Direction' Checkboxgroup component
				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
								# str (filepath or URL to image)
								# in 'Image' Image component
				"Howdy!",	# str in 'Inpaint Additional Prompt' Textbox component
				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
								# str (filepath or URL to image)
								# in 'Mask Upload' Image component
				True,	# bool in 'Disable Preview' Checkbox component
				True,	# bool in 'Disable Intermediate Results' Checkbox component
				True,	# bool in 'Disable seed increment' Checkbox component
				True,	# bool in 'Black Out NSFW' Checkbox component
				0.1,	# int | float (numeric value between 0.1 and 3.0)
								# in 'Positive ADM Guidance Scaler' Slider component
				0.1,	# int | float (numeric value between 0.1 and 3.0)
								# in 'Negative ADM Guidance Scaler' Slider component
				0,	# int | float (numeric value between 0.0 and 1.0)
								# in 'ADM Guidance End At Step' Slider component
				1,	# int | float (numeric value between 1.0 and 30.0)
								# in 'CFG Mimicking from TSNR' Slider component
				1,	# int | float (numeric value between 1 and 12)
								# in 'CLIP Skip' Slider component
				"euler",	# str (Option from: ['euler', 'euler_ancestral', 'heun', 'heunpp2', 'dpm_2', 'dpm_2_ancestral', 'lms', 'dpm_fast', 'dpm_adaptive', 'dpmpp_2s_ancestral', 'dpmpp_sde', 'dpmpp_sde_gpu', 'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu', 'ddpm', 'lcm', 'tcd', 'ddim', 'uni_pc', 'uni_pc_bh2'])
								# in 'Sampler' Dropdown component
				"normal",	# str (Option from: ['normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform', 'lcm', 'turbo', 'align_your_steps', 'tcd'])
								# in 'Scheduler' Dropdown component
				"Default (model)",	# str (Option from: ['Default (model)'])
								# in 'VAE' Dropdown component
				-1,	# int | float (numeric value between -1 and 200)
								# in 'Forced Overwrite of Sampling Step' Slider component
				-1,	# int | float (numeric value between -1 and 200)
								# in 'Forced Overwrite of Refiner Switch Step' Slider component
				-1,	# int | float (numeric value between -1 and 2048)
								# in 'Forced Overwrite of Generating Width' Slider component
				-1,	# int | float (numeric value between -1 and 2048)
								# in 'Forced Overwrite of Generating Height' Slider component
				-1,	# int | float (numeric value between -1 and 1.0)
								# in 'Forced Overwrite of Denoising Strength of "Vary"' Slider component
				-1,	# int | float (numeric value between -1 and 1.0)
								# in 'Forced Overwrite of Denoising Strength of "Upscale"' Slider component
				True,	# bool in 'Mixing Image Prompt and Vary/Upscale' Checkbox component
				True,	# bool in 'Mixing Image Prompt and Inpaint' Checkbox component
				True,	# bool in 'Debug Preprocessors' Checkbox component
				True,	# bool in 'Skip Preprocessors' Checkbox component
				1,	# int | float (numeric value between 1 and 255)
								# in 'Canny Low Threshold' Slider component
				1,	# int | float (numeric value between 1 and 255)
								# in 'Canny High Threshold' Slider component
				"joint",	# str (Option from: ['joint', 'separate', 'vae'])
								# in 'Refiner swap method' Dropdown component
				0,	# int | float (numeric value between 0.0 and 1.0)
								# in 'Softness of ControlNet' Slider component
				True,	# bool in 'Enabled' Checkbox component
				0,	# int | float (numeric value between 0 and 2)
								# in 'B1' Slider component
				0,	# int | float (numeric value between 0 and 2)
								# in 'B2' Slider component
				0,	# int | float (numeric value between 0 and 4)
								# in 'S1' Slider component
				0,	# int | float (numeric value between 0 and 4)
								# in 'S2' Slider component
				True,	# bool in 'Debug Inpaint Preprocessing' Checkbox component
				True,	# bool in 'Disable initial latent in inpaint' Checkbox component
				"None",	# str (Option from: ['None', 'v1', 'v2.5', 'v2.6'])
								# in 'Inpaint Engine' Dropdown component
				0,	# int | float (numeric value between 0.0 and 1.0)
								# in 'Inpaint Denoising Strength' Slider component
				0,	# int | float (numeric value between 0.0 and 1.0)
								# in 'Inpaint Respective Field' Slider component
				True,	# bool in 'Enable Mask Upload' Checkbox component
				True,	# bool in 'Invert Mask' Checkbox component
				-64,	# int | float (numeric value between -64 and 64)
								# in 'Mask Erode or Dilate' Slider component
				True,	# bool in 'Save Metadata to Images' Checkbox component
				"fooocus",	# str in 'Metadata Scheme' Radio component
				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
								# str (filepath or URL to image)
								# in 'Image' Image component
				0,	# int | float (numeric value between 0.0 and 1.0)
								# in 'Stop At' Slider component
				0,	# int | float (numeric value between 0.0 and 2.0)
								# in 'Weight' Slider component
				"ImagePrompt",	# str in 'Type' Radio component
				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
								# str (filepath or URL to image)
								# in 'Image' Image component
				0,	# int | float (numeric value between 0.0 and 1.0)
								# in 'Stop At' Slider component
				0,	# int | float (numeric value between 0.0 and 2.0)
								# in 'Weight' Slider component
				"ImagePrompt",	# str in 'Type' Radio component
				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
								# str (filepath or URL to image)
								# in 'Image' Image component
				0,	# int | float (numeric value between 0.0 and 1.0)
								# in 'Stop At' Slider component
				0,	# int | float (numeric value between 0.0 and 2.0)
								# in 'Weight' Slider component
				"ImagePrompt",	# str in 'Type' Radio component
				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# str (filepath or URL to image)
								# in 'Image' Image component
				0,	# int | float (numeric value between 0.0 and 1.0)
								# in 'Stop At' Slider component
				0,	# int | float (numeric value between 0.0 and 2.0)
					# in 'Weight' Slider component
				"ImagePrompt",	# str in 'Type' Radio component
				fn_index=46,
)
result = client.predict(fn_index=47)
print(result)