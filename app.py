
import streamlit as st
import os
import torch
import time

from e2e import predict_one_sample
from module.model import MT5PForSequenceClassification
from module.tokenizer import T5PegasusTokenizer

st.set_page_config(page_title="Demo", initial_sidebar_state="auto", layout="wide")


@st.cache(allow_output_mutation=True)
def get_model(device, vocab_path, model_path):
    tokenizer = T5PegasusTokenizer.from_pretrained(vocab_path)
    model = MT5PForSequenceClassification(model_path)
    #model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return tokenizer, model


device_ids = 7
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids)
device = torch.device("cuda" if torch.cuda.is_available() and int(device_ids) >= 0 else "cpu")
tokenizer, model = get_model(device, "t5_pegasus_torch/vocab.txt", "t5_pegasus_torch/")


def writer():
    st.markdown(
        """
        ## 摘要demo
        """
    )
    st.sidebar.subheader("配置参数")
    max_length = st.sidebar.slider("生成摘要长度", min_value=50, max_value=250, value=200, step=1)
    top_k = st.sidebar.slider("top_k", min_value=0, max_value=10, value=3, step=1)
    num_beams = st.sidebar.slider("num_beams", min_value=1, max_value=10, value=3, step=1)
    top_p = st.sidebar.number_input("top_p", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    do_sample = st.sidebar.checkbox('do_sample')
    content = st.text_area("输入新闻正文", max_chars=1024,height=400)
    if st.button("一键生成摘要"):
        start_message = st.empty()
        start_message.write("正在抽取，请等待...")
        start_time = time.time()
        title = predict_one_sample(model, device, tokenizer, content, max_length=max_length,do_sample=do_sample,
                                       num_beams=num_beams, top_k=top_k, top_p=top_p)
        end_time = time.time()
        start_message.write("抽取完成，耗时{}s".format(end_time - start_time))
        st.text_area("摘要如下",title)
        st.markdown(
            """
            ## 与基准系统T5生成的摘要性能比较
            """
        )
        col1, col2, col3,col4,col5 = st.columns(5)
        col1.metric("Rouge-1", "48.5", "16%")
        col2.metric("Rouge-2", "24.6", "-8%")
        col3.metric("Rouge-L", "34.9", "4%")
        col4.metric("BLEU", "24.0", "0%")
        col5.metric("BertScore", "64.7", "-3%")
    else:
        st.stop()

if __name__ == '__main__':
    writer()