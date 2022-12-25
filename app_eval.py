import json
import os
import streamlit as st
import time
from evaluate import Evaluator

st.set_page_config(page_title="Evaluate", initial_sidebar_state="auto", layout="wide")


@st.cache(allow_output_mutation=True)
def get_evaluator():
    evaluator = Evaluator()
    return evaluator


evaluator = get_evaluator()
device_ids = 7
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids)

def get_sources_targets(baseline_data):
    objects=json.loads(baseline_data)
    sources = objects["sources"]
    targets = objects["targets"]
    return sources, targets


def compute_diff(baselines, system):
    results = zip(baselines, system)
    diff_list = []
    for result in results:
        diff = round((result[1] - result[0]) / result[0], 2)
        diff_list.append(diff)
    return diff_list


def set_metric(container, baselines, system=None):
    col_name_list = ["Rouge-1", "Rouge-2", "Rouge-L", "BLEU", "BertScore"]
    cols = container.columns(5)
    if system != None:
        diff_list = compute_diff(baselines, system)
        for i in range(5):
            cols[i].metric(col_name_list[i], str(round(system[i],4)), str(diff_list[i]) + "%")
    else:
        for i in range(5):
            cols[i].metric(col_name_list[i], str(round(baselines[i],4)))


def writer():
    st.markdown(
        """
        ## 摘要评估
        """
    )
    st.sidebar.subheader("上传/下载")
    st.sidebar.write("请上传基准系统文件")
    baseline_uploaded_file = st.sidebar.file_uploader("基准系统")
    uploaded_files = st.sidebar.file_uploader("测试文件", accept_multiple_files=True)
    if st.button("一键评估"):
        start_message = st.empty()
        start_message.write("正在评估，请等待...")
        start_time = time.time()
        baseline_data = baseline_uploaded_file.read().decode('UTF-8')
        sources, targets = get_sources_targets(baseline_data)
        baseline_performance = evaluator.compute_all_score(sources, targets)
        baseline_container = st.container()
        baseline_container.write("基准系统性能表现")
        set_metric(baseline_container, baseline_performance)
        for index, uploaded_file in enumerate(uploaded_files):
            bytes_data = uploaded_file.read().decode('UTF-8')
            sources, targets = get_sources_targets(bytes_data)
            system_performance = evaluator.compute_all_score(sources, targets)
            container = st.container()
            container.write(uploaded_file.name + "系统的性能表现")
            set_metric(container, baseline_performance, system_performance)
        end_time = time.time()
        start_message.write("评估完成，耗时{}s".format(end_time - start_time))
    else:
        st.stop()


if __name__ == '__main__':
    writer()
