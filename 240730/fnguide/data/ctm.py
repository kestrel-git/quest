# -*- coding: utf-8 -*-
# %%

# %%
import re
import tomotopy as tp 
import io
from pyvis.network import Network # 네트워크 시각화에 사용할 패키지
import pickle as pk
import pandas as pd
import os
from datetime import datetime
import numpy as np
from konlpy.tag import Mecab

# %%
def extract_nouns(row):
    '''
    텍스트에서 명사를 추출하고,
    라이브러리에서 사용할 수 있는 형태로 가공
    @param text - 대상 텍스트 
    @return 단어로 구성된 라이브러리용 텍스트 
    '''
    # print(text)
    mecab = Mecab()

    try: 
        tokens = mecab.pos(row['TEXT'])
    except Exception as ex:
        tokens = []

    return " ".join([(w.upper()) for w,t in tokens if t[:2] in('NN', 'SL') and t!='NNBC' and t!='NNB'])



class CTM():
    __config = {} # topic_num : 기본 토픽 갯수, train_num : 기본 트래인 횟수, cur_path : CTM 모듈 리로드 경로, create_type : C - 신규생성, L - 기존 모듈 로딩
    __datas = None
    __stop_words_regex = ""
    __model_var = {}
    __cur_mdl = None #최근 모델 
    __procd_docs = [] #단어 리스트화된 문서 리스트 

    def __init__(self, config):
        self.__config = config

        if not os.path.exists(self.__config['cur_path'])  :
            os.makedirs(self.__config['cur_path'])

        if self.__config['create_type'] == 'L':
            self.load()

    
    def save(self):
        '''
        클래스 파일을 저장한다 
        '''
        config = self.__config

        with open("{0}/stop_words_regx.pkl".format(config['cur_path']), "wb") as f:
            pk.dump(self.__stop_words_regex, f)

        self.__make_mdl_bin__({ "mdl_bin_path": "{0}/mdl.bin".format(config['cur_path'])})
        self.__datas.to_pickle("{0}/data.pkl".format(config['cur_path']))
    
    def load(self):
        '''
        객체를 복구한다 
        '''
        config = self.__config
        with open("{0}/stop_words_regx.pkl".format(config['cur_path']), "rb") as f:
            self.__stop_words_regex = pk.load(f)

        self.__cur_mdl = tp.CTModel.load("{0}/mdl.bin".format(config['cur_path']))
        self.__datas = pd.read_pickle("{0}/data.pkl".format(config['cur_path']))

    def load_data(self, df_datas, stop_words):
        '''
        분석할 데이터를 로드함
        : df_datas pandas dataframe 형의 데이터 필수 컬럼 : CLN_TXT
        '''
        self.__datas = df_datas.copy()
        self.__make_stop_words_regx__(stop_words)

        pcd_doc = []
        for i, row in self.__datas.iterrows():
            tmp_doc = row['CLN_TXT']

            if self.__stop_words_regex != None:
                tmp_doc = tmp_doc.replace("/", "//").replace("|", "||")
                tmp_doc = re.sub(self.__stop_words_regex, r"\g<1>", tmp_doc, flags=re.IGNORECASE)
                tmp_doc = tmp_doc.replace("//", "/").replace("||", "|")

            tmp_doc = re.sub(r"/|\|"," ",tmp_doc)
            tmp_doc = tmp_doc.strip()

            if tmp_doc != "":
                pcd_doc.append(tmp_doc.split()) 
            else:
                pcd_doc.append(['']) 

        self.__procd_docs = pcd_doc

        #if stop_words != None:

   
    def load_data_v2(self, df_datas, stop_words):
        '''
        분석할 데이터를 로드함 

        : df_datas pandas dataframe 형의 데이터 필수 컬럼 : TEXT
        '''
        self.__datas = df_datas.copy()
        self.__make_stop_words_regx__(stop_words)

        self.__datas['PRE_PROC_TXT'] = self.__datas.apply(extract_nouns, axis=1)

        pcd_doc = []
        for i, row in self.__datas.iterrows():
            tmp_doc = row['PRE_PROC_TXT']

            if self.__stop_words_regex != None:
                tmp_doc = re.sub(self.__stop_words_regex, r"\g<1>", tmp_doc, flags=re.IGNORECASE)

            tmp_doc = tmp_doc.strip()

            if tmp_doc != "":
                pcd_doc.append(tmp_doc.split()) 
            else:
                pcd_doc.append(['']) 

        self.__procd_docs = pcd_doc


    def __make_stop_words_regx__(self, stop_words):
        '''
        불용어 정규 표현식 생성
        : 불용 리스트 
        '''
        if stop_words != None:
            self.__stop_words_regex = "(^|/|\|)({0})($|/|\|)".format("|".join([ re.escape(sw) for sw in stop_words]))
        else:
            self.__stop_words_regex = None

    def get(self, info_nm):
        res_info = None

        if info_nm == 'MDL':
            res_info = self.__cur_mdl

        return res_info

    def simulate(self, options={}):
        '''
        토픽 모델링을 수행함 
        : options 모델링 옵션 topic_num - 추출 토픽 갯수, train_num - 훈련횟수 
        '''
        res_state = 0

        try :
            for opt_nm in ['topic_num', 'train_num']:
                if 'topic_num' in options:
                    self.__model_var[opt_nm] = int(options[opt_nm])
                else:
                    self.__model_var[opt_nm] = int(self.__config[opt_nm])

            mdl = tp.CTModel(k=self.__model_var['topic_num'])

            for doc in self.__procd_docs:
                mdl.add_doc(doc)

            mdl.train(iter=self.__model_var['train_num'])

            self.__cur_mdl = mdl
        except:
            self.__cur_mdl = None
            res_state = -1
        
        return res_state

    def extract(self, item_nm, param):
        '''
        정보를 추출함 
        : @param item_nm - CHART : 현재 모델 차트 HTML 생성(chart_path : 차트 생성경로)
        :                  MODEL : 모델 BIN 생성(mdl_bin_path : 모델 바이너리 경로)
        '''
        if item_nm == 'CHART':
            return self.__make_chart__(param)
        elif item_nm == 'MODEL':
            return self.__make_mdl_bin__(param)

    def __make_chart__(self, param):
        res_state  = 0
        mdl = self.__cur_mdl
        
        try:
            g = Network(width=2400, height=1600, font_color="#333")
            
            correl = mdl.get_correlations().reshape([-1])
            correl.sort()
            
            top_tenth = mdl.k * (mdl.k - 1) // 10
            top_tenth = correl[-mdl.k - top_tenth]
            
            topic_counts = mdl.get_count_by_topics()
            
            for k in range(mdl.k):
                label = "#{}".format(k)
                title= ' '.join(word for word, _ in mdl.get_topic_words(k, top_n=8))
                label += '\n' + ' '.join(word for word, _ in mdl.get_topic_words(k, top_n=3))
                g.add_node(k, label=label, title=title, shape='ellipse', value=float(topic_counts[k]))
                for l, correlation in zip(range(k - 1), mdl.get_correlations(k)):
                    if correlation < top_tenth: continue
                    g.add_edge(k, l, value=float(correlation), title='{:.02}'.format(correlation))
            
            g.barnes_hut(gravity=-8000, spring_length=20)
            g.write_html(param['chart_path'])
        except  Exception as ex:
            print(ex)
            res_state = -1

        return res_state

    def __make_mdl_bin__(self, param):
        res_state  = 0

        try:
            self.__cur_mdl.save(param['mdl_bin_path'], full=True)
        except  Exception as ex:
            print(ex)
            res_state = -1
        
        return res_state


# %%
class ThemeInferencer():
    '''
    테마 추정기 
    '''
    _df_theme = None #테마-키워드 비중 데이터 
    _mdl = None #모델데이터  
    _infer_limit = 2 #추정 토픽수
    _infer_summary_word_limit = 2 #핵심단어
    _tpc_word_top_n = 15 #토픽 N 단어
    _excel_path = '' #엑셀 경로

    def __init__(self, mdl, df_theme, excel_dir):
        self._mdl = mdl
        self._tpc_n = mdl.get_count_by_topics().shape[0]
        self._df_theme = df_theme
        self._excel_dir = excel_dir

        self.__set_arr_thm_key_val__()
        self.__set_tpc_word_dist__()
        self.__infer_tpc_thm__()

    def __set_arr_thm_key_val__(self):
        '''
        테마마스터를 토픽수X테마키워드 가중치(토픽모델링 단어수) 형태의 백터로 변환
        '''
        df_thm_kwd =  self._df_theme
        theme_nms = df_thm_kwd['NAME'].unique()
        mdl_vocabs = np.array(self._mdl.vocabs)
        arr_thm_key_val = np.zeros((theme_nms.shape[0], mdl_vocabs.shape[0]), dtype=np.float)

        cur_theme_idx = 0
        for theme_nm in theme_nms:
            tmp_thm_kwd = df_thm_kwd[df_thm_kwd['NAME'] == theme_nm]
            arr_tmp_kwd_val = tmp_thm_kwd[['WORD', 'WEIGHT']].to_numpy()
            #print('NAME : {0}'.format(theme_nm))
            
            for tmp_kwd, val in arr_tmp_kwd_val:
                tmp_res = np.where(mdl_vocabs == tmp_kwd) #포함배열인덱스, 인덱스형 순서쌍
                if tmp_res[0].shape[0] > 0:
                    arr_thm_key_val[cur_theme_idx, tmp_res[0][0]] = val
            
            cur_theme_idx = cur_theme_idx + 1

        self._arr_thm_key_val = arr_thm_key_val
        self._df_thm_nms = theme_nms

    def __set_tpc_word_dist__(self):
        '''
        토픽인덱스 기준 토픽수*단어별 출현확률(단어수) 백터로 변환
        '''
        tpc_wrd_dist = []
        arr_tpc_wrd_dist = None

        for idx in range(0, self._tpc_n):
            tpc_wrd_dist.append(self._mdl.get_topic_word_dist(idx))
        arr_tpc_wrd_dist = np.array(tpc_wrd_dist)

        self._arr_tpc_wrd_dist = arr_tpc_wrd_dist

    def __infer_tpc_thm__(self):
        '''
        토픽별 테마 추정 및 토픽별 핵심단어 선별 
        _tpc_infer_thms(토픽수, 추정테마수) _tpc_summary_words_idx(토픽수, 추정테마수-테마인덱스, 핵심단어수-단어인덱스)
        '''
        tpc_thm_score = np.dot(self._arr_tpc_wrd_dist, np.transpose(self._arr_thm_key_val)) #토픽별 추정 테마
        self._tpc_thm_score = tpc_thm_score
        self._tpc_infer_thms = np.argsort(tpc_thm_score * -1, axis=1)[:, :self._infer_limit] #추정 토픽 
        tpc_summary_words_idx = []
        for idx in range(0, self._arr_tpc_wrd_dist.shape[0]):
            tpc_summary_words_idx.append(
                np.argsort(self._arr_tpc_wrd_dist[idx] * self._arr_thm_key_val[self._tpc_infer_thms[idx]] * -1)[:,:self._infer_summary_word_limit]
            )
        self._tpc_summary_words_idx = np.array(tpc_summary_words_idx) #토픽별 핵심단어

    def __make_res_view__(self):
        '''
        추출된 토픽과 유사한 보유테마 뷰를 만든다. 
        토픽인덱스/토픽단어와 단어확률/유사 보유테마 인덱스/보유 테마명/매칭 핵심단어

        @return 데이터 프레임(토픽인덱스/토픽단어와 단어확률/유사 보유테마 인덱스/보유 테마명/매칭 핵심단어)
        '''
        
        tmp_res_infer_view = []
        for tpc_idx in range(0, self._tpc_n):
            tmp_tpc_words_txt = []
            for word, dist in self._mdl.get_topic_words(topic_id=tpc_idx, top_n=self._tpc_word_top_n):
                tmp_tpc_words_txt.append("{0}({1})".format(word, round(dist, 4)))
            for inf_thm_idx in  range(0, self._infer_limit):
                tmp_words = []
                for wrd_idx in range(0, self._infer_summary_word_limit):
                    tmp_words.append(self._mdl.vocabs[self._tpc_summary_words_idx[tpc_idx, inf_thm_idx, wrd_idx]])

                tmp_res_infer_view.append([tpc_idx, "/".join(tmp_tpc_words_txt), inf_thm_idx, self._df_thm_nms[ self._tpc_infer_thms[tpc_idx, inf_thm_idx]], "/".join(tmp_words)])
        
        df = pd.DataFrame(tmp_res_infer_view, columns=['TOPIC_IDX','TOPIC_WORDS','INF_THM_IDX', 'INF_THM_NM', 'INF_THM_WORDS'])
        
        #df_res_thm = pd.DataFrame(self._df_theme[self._df_theme['THM_WD_RNK'] < 6])
        df_res_thm = self._df_theme
        df_res_thm['WD_EX'] = self._df_theme['WORD'] +'('+ round(self._df_theme['WEIGHT'],4).astype(str) + ')'
        df_res_thm['WDS_EX'] = df_res_thm[['NAME','WD_EX']].groupby(['NAME']).transform(lambda x : '/'.join(x))
        df_res_thm = df_res_thm[['NAME', 'WDS_EX']].drop_duplicates()
        
        df = pd.merge(df, df_res_thm, how='left', left_on=['INF_THM_NM'], right_on=['NAME'])
        df = df[['TOPIC_IDX', 'TOPIC_WORDS', 'INF_THM_IDX', 'INF_THM_NM', 'WDS_EX', 'INF_THM_WORDS']]

        return df


    def get(self):
        res_v = self.__make_res_view__()

        return res_v
