#!/usr/bin/env python
__author__ = 'solivr'
__licence__ = 'GPL'

import pandas as pd

_accents_list = 'àéèìîóòù'
_accent_mapping = {'à': 'a',
                   'é': 'e',
                   'è': 'e',
                   'ì': 'i',
                   'î': 'i',
                   'ó': 'o',
                   'ò': 'o',
                   'ù': 'u'}


def map_accentuated_characters_in_dataframe(dataframe_transcriptions: pd.DataFrame,
                                            dict_mapping: dict=_accent_mapping) -> pd.DataFrame:
    """

    :param dataframe_transcriptions: must have a field 'transcription'
    :param dict_mapping
    :return:
    """
    items = dataframe_transcriptions.transcription.iteritems()

    for i in range(dataframe_transcriptions.transcription.count()):
        df_id, transcription = next(items)
        # https://stackoverflow.com/questions/30020184/how-to-find-the-first-index-of-any-of-a-set-of-characters-in-a-string
        ch_index = next((i for i, ch in enumerate(transcription) if ch in _accents_list), None)
        while ch_index is not None:
            transcription = list(transcription)
            ch = transcription[ch_index]
            transcription[ch_index] = dict_mapping[ch]
            transcription = ''.join(transcription)
            dataframe_transcriptions.at[df_id, 'transcription'] = transcription
            ch_index = next((i for i, ch in enumerate(transcription) if ch in _accents_list), None)

    return dataframe_transcriptions


def map_accentuated_characters_in_string(string_to_format: str, dict_mapping: dict=_accent_mapping) -> str:
    """

    :param string_to_format:
    :param dict_mapping:
    :return:
    """
    # https://stackoverflow.com/questions/30020184/how-to-find-the-first-index-of-any-of-a-set-of-characters-in-a-string
    ch_index = next((i for i, ch in enumerate(string_to_format) if ch in _accents_list), None)
    while ch_index is not None:
        string_to_format = list(string_to_format)
        ch = string_to_format[ch_index]
        string_to_format[ch_index] = dict_mapping[ch]
        string_to_format = ''.join(string_to_format)
        ch_index = next((i for i, ch in enumerate(string_to_format) if ch in _accents_list), None)

    return string_to_format


def format_string_for_tf_split(string_to_format: str, separator_character: str= '|',
                               replace_brackets_abbreviations=True) -> str:
    """
    Formats transcriptions to be split by tf.string_split using character separator "|"
    :param string_to_format: string to format
    :param separator_character: character that separates alphabet units
    :param replace_brackets_abbreviations: if True will replace '[' and ']' chars by separator character
    :return:
    """

    if replace_brackets_abbreviations:
        # Replace "[]" chars by "|"
        string_to_format = string_to_format.replace("[", separator_character).replace("]", separator_character)
    # Split words
    tokens = string_to_format.split(' ')

    final_string = ''
    for tk in tokens:
        if tk[0] == separator_character:  # Case for abbreviation
            final_string += '{} '.format(tk)
        else:  # Case for 'standard' word
            final_string += '{} '.format(tk.replace('', separator_character))
    return final_string[:-1]


def lower_abbreviation_in_string(string_to_format: str):
    # Split with '['
    tokens_opening = string_to_format.split('[')

    valid_string = True
    final_string = tokens_opening[0]
    for tok in tokens_opening[1:]:
        if len(tok) > 1:
            token_closing = tok.split(']')
            if len(token_closing) == 2:  # checks if abbreviation starts with [ and ends with ]
                final_string += '[' + token_closing[0].lower() + ']' + token_closing[1]
            else:  # No closing ']'
                valid_string = False
        else:
            final_string += ']'
    if valid_string:
        return final_string
    else:
        return ''
