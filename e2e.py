import torch
from torch.nn.utils.rnn import pad_sequence


def get_input(input_text, tokenizer):
    input_ids_list = []
    input_ids = [tokenizer.cls_token_id]
    text_tokens = tokenizer.tokenize(input_text)
    text_tokens = tokenizer.convert_tokens_to_ids(text_tokens)
    input_ids.extend(text_tokens)
    input_ids.append(tokenizer.sep_token_id)
    input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    return input_ids


def generate_results(output, tokenizer):
    title_list = []
    for i in range(len(output)):
        title = ''.join(tokenizer.decode(output[i][1:], skip_special_tokens=True)).replace(' ', '')
        title_list.append(title)
    return title_list


def predict_one_sample(model, device, tokenizer, input_text, max_length=200, do_sample=False, num_beams=5, top_k=50,
                       top_p=3):
    input_ids = get_input(input_text, tokenizer)
    input_ids = input_ids.to(device)
    text_output = model.generate(input_ids, decoder_start_token_id=tokenizer.cls_token_id,
                                 eos_token_id=tokenizer.sep_token_id, max_length=max_length, do_sample=do_sample,
                                 num_beams=num_beams, top_k=top_k, top_p=top_p).cpu().numpy()
    title_list = generate_results(text_output, tokenizer)
    return title_list[0]
