from sty import bg
import numpy as np
def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{}\033[38;2;255;255;255m".format(r, g, b, text)

def colored_bg(r, g, b, text):
    return bg(r, g, b) + text + bg.rs

def color_by_importance(importance, gt, word):
    v_1 = int(255*(1-importance))
    if gt == 1:
        return colored(40, 181, 40, colored_bg(255, v_1, v_1, word))
    else:
        return colored(0, 0, 0, colored_bg(255, v_1, v_1, word))

def visualize_word_importance(words):
    if len(words[0]) == 2:
        words = [(w[0], 0, w[1]) for w in words]
    current_line_length = 0
    for i in range(len(words)):
        if current_line_length + len(words[i][2]) + 1 > 180:
            if i != (len(words)-1):
                print(color_by_importance(0, 0, " "*(180-current_line_length)), end="")
            print()
            print(colored(0, 0, 0, color_by_importance(*words[i])), end="")
            current_line_length = len(words[i][2]) + 1
        else:
            if i > 0:
                print(color_by_importance((words[i-1][0] + words[i][0])/2, (words[i-1][1] + words[i][1])/2, " "), end="")
            print(colored(0, 0, 0, color_by_importance(*words[i])), end="")
            current_line_length += len(words[i][2]) + 1
        if current_line_length > 180:
            print()
            current_line_length = 0

    print()