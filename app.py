import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import Union

import sys
import gradio as gr

from src.data.components.vocab import Vocab
from src.models.components.corrector import Corrector
from src.models.components.model import ModelWrapper
from src.models.components.util import load_weights
from src.data.components.noise import SynthesizeData
from src.utils.api_utils import correctFunction, postprocessing_result


model_name = "tfmwtr"
dataset = "dxl"
vocab_path = f'data/{dataset}/{dataset}.vocab.pkl'
# weight_path = f'data/checkpoints/tfmwtr/{dataset}.weights.pth'
weight_path = f'data/checkpoints/tfmwtr/last.ckpt'
vocab = Vocab("vi")
vocab.load_vocab_dict(vocab_path)
noiser = SynthesizeData(vocab)
device = 'cuda'
model_wrapper = ModelWrapper(f"{model_name}", dataset, device)
corrector = Corrector(model_wrapper)
load_weights(corrector.model, weight_path)

def correct(string: str):
    out = correctFunction(string, corrector)
    result = postprocessing_result(out)

    ret = []
    for r in result:
        r = [s.strip() for s in r if isinstance(s, str)]

        if len(r) == 2:
            ret.append((r[0], r[1]))
        else:
            ret.append((r[0], None))
        ret.append((" ", None))
    ret.pop()
    print(ret, "RET")
    return ret

if __name__ == "__main__":
    css = """
    #output {
        .label {
            background-color: green !important;
        }
    }
    """
    gr.Interface(
        correct,
        inputs=gr.Textbox(label="Input", placeholder="Enter text to be corrected here..."),
        outputs=gr.HighlightedText(
            label="Output",
            combine_adjacent=True,
            show_label=True,
            elem_id="output"
        ),
        theme=gr.themes.Default(),
        css=css,
        examples=[
            "Việ nâng cảo trình độ quản lý, nâng cao hiểu biết về kinh doanh hiện đã tr73 thành nhu cầu thiết yếu với đổi ngũ nhan viên, cán bộ trẻ và các nhà lãnh đạo doanh nghiệp trong bốj cảnh Việ Nam đang chuẩn bị gia nhập WTO và hội nhiếp về nhiều mặt với thế giới.. Thực tế cho thấy, những người đạt bằng MBA có sự vững vàng về kiến thức kinh doanh cũng như có nhiều cơ hội phăng tiến hơn troxg cuộng việc và cuộc sống.",
            "Bề ngoài,p trông chúng giống như một con ma cà rồng trong trof chơi ở lễ hoi Halloween.",
            "Nuoc mat em roi tro choi ket thuc",
            "Dai hoc Cong Nghệ, Dai hocquoc Gia Há Noj",
        ]
    ).launch(share=True)