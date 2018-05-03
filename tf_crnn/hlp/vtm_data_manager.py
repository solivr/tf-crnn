#!/usr/bin/env python
__author__ = 'solivr'

import pandas as pd

accents_list = 'àéèìîóòù'
accent_mapping = {'à': 'a',
                  'é': 'e',
                  'è': 'e',
                  'ì': 'i',
                  'î': 'i',
                  'ó': 'o',
                  'ò': 'o',
                  'ù': 'u'}


def map_accentuated_characters(dataframe_transcriptions, accent_mapping):
    """

    :param dataframe: must have a field 'transcription'
    :return:
    """
    items = dataframe_transcriptions.transcription.iteritems()

    for i in range(dataframe_transcriptions.transcription.count()):
        df_id, transcription = next(items)
        # https://stackoverflow.com/questions/30020184/how-to-find-the-first-index-of-any-of-a-set-of-characters-in-a-string
        ch_index = next((i for i, ch in enumerate(transcription) if ch in accents_list), None)
        while ch_index is not None:
            transcription = list(transcription)
            ch = transcription[ch_index]
            transcription[ch_index] = accent_mapping[ch]
            transcription = ''.join(transcription)
            dataframe_transcriptions.at[df_id, 'transcription'] = transcription
            ch_index = next((i for i, ch  in enumerate(transcription) if ch in accents_list), None)

    return dataframe_transcriptions