import re
import string

def normalize(sentence:str)->str:
    """
    normalize the answer to increase the score
    """
    retokenization_rules = [
    # Remove extra space around single quotes, hyphens, and slashes.
    (" ' (.*?) ' ", " '\\1' "),
    (" - ", "-"),
    (" / ", "/"),
    # Ensure there are spaces around parentheses and brackets.
    (r"([\]\[\(\){}<>])", " \\1 "),
    (r"\s+", " "),
    ]
    for rule in retokenization_rules:
        line = re.sub(rule[0], rule[1], sentence)
    return line

def extract_model_response(response_text:str)->str:
    """
    extract the first round of conversation
    """
    # Define the start and end markers for the model's response
    start_marker = "<<SYS>>"
    end_marker = "<</SYS>>"
    
    # Find the first occurrence of start_marker and end_marker
    start_index = response_text.find(start_marker)
    end_index = response_text.find(end_marker, start_index)
    
    # Extract the text between the start index of the start_marker and the end index of the end_marker
    if start_index != -1 and end_index != -1:
        # Add the length of start_marker to start index to skip the marker itself
        model_response = response_text[start_index + len(start_marker):end_index].strip()
        return model_response
    else:
        return None

def remove_before_first_inst(text:str)->str:
    # Find the position of the first occurrence of [/INST]
    inst_pos = text.find("[/INST]")
    if inst_pos != -1:
        # Remove everything before (and including) the first [/INST]
        return text[inst_pos + len("[/INST]"):]
    else:
        # If [/INST] is not found, return the original text
        return None
    
def clean_sentence(sentence):
    sentence = sentence.replace('Corrected: ', '')
    if sentence.endswith('"'):
        sentence = sentence.rsplit('"', 1)[0].strip()
    return sentence

def clean_punctuation(text):
    punctuation = (string.punctuation)
    punctuation+=" `"
    punctuation = set(punctuation)
    if len(text) == 0:
        return text
    end = len(text)
    for i in range(len(text) - 1, -1, -1):
        if text[i] in punctuation:
            end = i
        else:
            break
    return text[:end+2]


def retain_content_before_last_non_english_punctuation(text):
    # Reverse iterate over the text from the end
    for i in range(len(text) - 1, 0, -1):
        # Check if the current character is a space and the previous one is an English letter
        if text[i] == ' ' and text[i-1].isalpha():
            # Return the text up to the space
            return text[:i+2]
    # Return the original text if no such space is found
    return text
