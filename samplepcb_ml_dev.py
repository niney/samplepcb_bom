#!/usr/bin/env python
# coding: utf-8

# In[1]:



import logging
from logging import handlers

import numpy as np

# import sentence_hub as sh

import column_analysis as colanal

logger = None


# In[2]:


def make_logger(name=None):
    #1 logger instance를 만든다.
    logger = logging.getLogger(name)

    #2 logger의 level을 가장 낮은 수준인 DEBUG로 설정해둔다.
    logger.setLevel(logging.DEBUG)

    #3 formatter 지정
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    #4 handler instance 생성
    console = logging.StreamHandler()
    file_handler = handlers.TimedRotatingFileHandler(filename='app.log', when='midnight', interval=1, encoding='utf-8')

    #5 handler 별로 다른 level 설정
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    #6 handler 출력 format 지정
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y%m%d"

    #7 logger에 handler 추가
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


# In[3]:


##### MAIN SCRIPT #####

if __name__ == '__main__':
    # try:
    logger = make_logger()
    logger.info("Downloading pre-trained embeddings from tensorflow hub...")
    # sh.init()
    logger.info("init complete!!!!!!")


# In[10]:


from imp import reload
reload(colanal)

# column_cnt_list, sheet = colanal.load_bom_excel('BOM_3_YONG.xlsx')
column_cnt_list, sheet = colanal.load_bom_excel('1GuardFin_1_70.xls')
# column_cnt_list, sheet = colanal.load_bom_excel('PartList_DRSD-Atype_A_TOP_Rev02A_r02.xlsx')
search_result = colanal.search_pcb_header_each(sheet)
colanal.bom_ml_init()
item_detail = colanal.search_pcb_column_each(sheet, search_result['headerColumnIdx'] + 1, search_result['headerDetail']['pcbColumnSearchList'])
item_detail2 = colanal.search_pcb_column_cols(sheet, search_result['headerColumnIdx'] + 1, search_result['headerDetail']['pcbColumnSearchList'])
print(item_detail2)
