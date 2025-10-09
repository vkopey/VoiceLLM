#encoding utf-8

import glob
import os
print("Поточний каталог:", os.getcwd()) 
import re
import gradio as gr

import spaces



from transformers import MBartForConditionalGeneration, AutoTokenizer

verbalizer_model_name = "skypro1111/mbart-large-50-verbalization"


class Verbalizer():
    def __init__(self, device):
        self.device = device

        self.model = MBartForConditionalGeneration.from_pretrained(verbalizer_model_name,
            low_cpu_mem_usage=True,
            device_map=device,
        )
        self.model.eval()
    
        self.tokenizer = AutoTokenizer.from_pretrained(verbalizer_model_name)
        self.tokenizer.src_lang = "uk_XX"
        self.tokenizer.tgt_lang = "uk_XX"
    
    def generate_text(self, text):
        """Generate text for a single input."""
        # Prepare input
        input_text = "<verbalization>:" + text

        encoded_input = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)
        output_ids = self.model.generate(
            **encoded_input, max_length=1024, num_beams=5, early_stopping=True
        )
        normalized_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


        return normalized_text.strip()

import torch
from ipa_uk import ipa
from unicodedata import normalize
from styletts2_inference.models import StyleTTS2
from ukrainian_word_stress import Stressifier, StressSymbol
stressify = Stressifier()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

prompts_dir = 'f:\\python-3.10.11-embed-amd64_TTS\\Scripts\\styletts2-ukrainian\\voices' #'voices'

verbalizer = Verbalizer(device=device)


def split_to_parts(text):
    split_symbols = '.?!:'
    parts = ['']
    index = 0
    for s in text:
        parts[index] += s
        if s in split_symbols and len(parts[index]) > 150:
            index += 1
            parts.append('')
    return parts



single_model = StyleTTS2(hf_path='patriotyk/styletts2_ukrainian_single', device=device)
single_style = torch.load('filatov.pt')


multi_model = StyleTTS2(hf_path='patriotyk/styletts2_ukrainian_multispeaker', device=device)
multi_styles = {}

prompts_list = sorted(glob.glob(os.path.join(prompts_dir, '*.pt')))
prompts_list = ['.'.join(p.split('/')[-1].split('.')[:-1]) for p in prompts_list]

for audio_prompt in prompts_list:
    audio_path = os.path.join(prompts_dir, audio_prompt+'.pt')
    multi_styles[audio_prompt] = torch.load(audio_path)
    print('loaded ', audio_prompt)

models = {
    'multi': {
        'model': multi_model,
        'styles': multi_styles
    },
    'single': {
        'model': single_model,
        'style': single_style
    }
}


@spaces.GPU
def verbalize(text):
    parts = split_to_parts(text)
    verbalized = ''
    for part in parts:
        if part.strip():
            verbalized += verbalizer.generate_text(part) + ' '
    return verbalized

description = f'''
<h1 style="text-align:center;">StyleTTS2 ukrainian demo</h1><br>
Програма може не коректно визначати деякі наголоси.
Якщо наголос не правильний, використовуйте символ + після наголошеного складу.
Текст який складається з одного слова може синтезуватися з певними артефактами, пишіть повноцінні речення.
Якщо текст містить цифри чи акроніми, можна натисну кнопку "Вербалізувати" яка повинна замінити цифри і акроніми
в словесну форму.

'''

examples = [
    ["Решта окупантів звернула на Вокзальну — центральну вулицю Бучі. Тільки уявіть їхній настрій, коли перед ними відкрилася ця пасторальна картина! Невеличкі котеджі й просторіші будинки шикуються обабіч, перед ними вивищуються голі липи та електро-стовпи, тягнуться газони й жовто-чорні бордюри. Доглянуті сади визирають із-поза зелених парканів, гавкотять собаки, співають птахи… На дверях будинку номер тридцять шість досі висить різдвяний вінок.", 1.0],
    ["Одна дівчинка стала королевою Франції. Звали її Анна, і була вона донькою Ярослава Му+дрого, великого київського князя. Він опі+кувався літературою та культурою в Київській Русі+, а тоді переважно про таке не дбали – більше воювали і споруджували фортеці.", 1.0],
    ["Одна дівчинка народилася і виросла в Америці, та коли стала дорослою, зрозуміла, що дуже любить українські вірші й найбільше хоче робити вистави про Україну. Звали її Вірляна. Дід Вірляни був український мовознавець і педагог Кость Кисілевський, котрий навчався в Лейпцизькому та Віденському університетах і, після Другої світової війни виїхавши до США, започаткував систему шкіл українознавства по всій Америці. Тож Вірляна зростала в українському середовищі, а окрім того – в середовищі вихідців з інших країн.", 1.0],
    ["За інформацією від Державної служби з надзвичайних ситуацій станом на 7 ранку 15 липня.", 1.0],
    ["Очікується, що цей застосунок буде запущено 22.08.2025.", 1.0],
]



@spaces.GPU
def synthesize(model_name, text, speed, voice_name = None, progress=gr.Progress()):
    
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if len(text) > 50000:
        raise gr.Error("Text must be <50k characters")
    print("*** saying ***")
    print(text)
    print("*** end ***")
    

    result_wav = []
    for t in progress.tqdm(split_to_parts(text)):

        t = t.strip()
        t = t.replace('"', '')
        if t:
            t = t.replace('+', StressSymbol.CombiningAcuteAccent)
            t = normalize('NFKC', t)

            t = re.sub(r'[᠆‐‑‒–—―⁻₋−⸺⸻]', '-', t)
            t = re.sub(r' - ', ': ', t)
            ps = ipa(stressify(t))

            if ps:
                tokens = models[model_name]['model'].tokenizer.encode(ps)
                if voice_name:
                    style = models[model_name]['styles'][voice_name]
                else:
                    style = models[model_name]['style']
                
                wav = models[model_name]['model'](tokens, speed=speed, s_prev=style)
                result_wav.append(wav)

    
    return 24000, torch.concatenate(result_wav).cpu().numpy()



def select_example(df, evt: gr.SelectData):
    return evt.row_value   
    
with gr.Blocks() as single:
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Text(label='Text:', lines=5, max_lines=10)
            verbalize_button = gr.Button("Вербалізувати(beta)")
            speed = gr.Slider(label='Швидкість:', maximum=1.3, minimum=0.7, value=1.0)
            verbalize_button.click(verbalize, inputs=[input_text], outputs=[input_text])
            
        with gr.Column(scale=1):
            output_audio = gr.Audio(
                    label="Audio:",
                    autoplay=False,
                    streaming=False,
                    type="numpy",
                )
            synthesise_button = gr.Button("Синтезувати")
            single_text = gr.Text(value='single', visible=False)
            synthesise_button.click(synthesize, inputs=[single_text, input_text, speed], outputs=[output_audio])
    
    with gr.Row():
        examples_table = gr.Dataframe(wrap=True, headers=["Текст", "Швидкість"], datatype=["str", "number"], value=examples, interactive=False)
        examples_table.select(select_example, inputs=[examples_table], outputs=[input_text, speed])
    
with gr.Blocks() as multy:
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Text(label='Text:', lines=5, max_lines=10)
            verbalize_button = gr.Button("Вербалізувати(beta)")
            speed = gr.Slider(label='Швидкість:', maximum=1.3, minimum=0.7, value=1.0)
            speaker = gr.Dropdown(label="Голос:", choices=prompts_list, value=prompts_list[0])
            verbalize_button.click(verbalize, inputs=[input_text], outputs=[input_text])

        with gr.Column(scale=1):
            output_audio = gr.Audio(
                    label="Audio:",
                    autoplay=False,
                    streaming=False,
                    type="numpy",
                )
            synthesise_button = gr.Button("Синтезувати")
            multi = gr.Text(value='multi', visible=False)
            
            synthesise_button.click(synthesize, inputs=[multi, input_text, speed, speaker], outputs=[output_audio])
    with gr.Row():
        examples_table = gr.Dataframe(wrap=True, headers=["Текст", "Швидкість"], datatype=["str", "number"], value=examples, interactive=False)
        examples_table.select(select_example, inputs=[examples_table], outputs=[input_text, speed])




with gr.Blocks(title="StyleTTS2 ukrainian demo", css="") as demo:
    gr.Markdown(description)
    gr.TabbedInterface([multy, single], ['Multі speaker', 'Single speaker'])
    

if __name__ == "__main__":
    demo.queue(api_open=True, max_size=15).launch(show_api=True, share=True)