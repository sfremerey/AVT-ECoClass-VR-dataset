#!/usr/bin/env python
# coding: utf-8

# # Initial data processing (ETL)

# In[168]:


# ipython
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import display as d
from IPython.display import Image

import pandas as pd
import pingouin as pg
import numpy as np
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ruptures as rpt
sns.set(rc={'figure.figsize':(5,5)})
sns.set(font_scale=0.9)
sns.set_style("white")
pd.set_option('display.max_columns', None)
import re
import glob
import json
import os
import sys
import json

from IPython.display import display as d
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests

# R stuff
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

def save_fig(ax, title, pdfname):
    ax.set_title(title)
    ax.get_figure().savefig(pdfname, bbox_inches="tight", dpi=300)

palette = sns.color_palette("deep")
sns.set_palette(palette)


# In[169]:


# avrateNG - Pre Simulator Sickness Questionnaire (SSQ) and Weinstein Noise Sensitivity Scale (WNSS)
test_ids = ["1", "2", "3"]
test_names = ["360_diotic", "360_binaural", "cgi_binaural"]
pre_ssq_df = pd.DataFrame()
weinstein_df = pd.DataFrame()

for test_id, test_name in zip(test_ids, test_names):
    df = pd.read_csv("../subjective_data/{}/avrateNG/_info.csv".format(test_name), index_col=0)

    info_df = pd.json_normalize(df['info_json'].apply(json.loads))
    info_df['subject_number'] = range(1, len(info_df) + 1)
    info_df['test_id'] = test_id
    columns_to_keep = ['subject_number', 'test_id', 'radio_allgemeines_unbehagen_bzw_unwohlsein', 'radio_ermuedung', 'radio_kopfschmerzen', 'radio_ueberanstrengte_augen', 'radio_schwierigkeiten_mit_sehschaerfe', 'radio_erhoehte_speichelbildung', 'radio_schwitzen', 'radio_uebelkeit__erbrechen_konzentrationsschwierigkeiten', 'radio_druckgefuehl_im_kopfbereich', 'radio_verschwommene_sicht', 'radio_schwindelgefuehl__bei_geoeffneten_augen_', 'radio_schwindelgefuehl__bei_geschlossenen_augen_', 'radio_gleichgewichtsstoerungen', 'radio_magenbeschwerden', 'radio_aufstossen']
    pre_ssq_df_tmp = info_df[columns_to_keep]
    pre_ssq_df_tmp = pre_ssq_df_tmp.rename(columns=lambda x: x.replace('radio_', ''))
    N_columns = ['allgemeines_unbehagen_bzw_unwohlsein', 'erhoehte_speichelbildung', 'schwitzen', 'uebelkeit__erbrechen_konzentrationsschwierigkeiten', 'uebelkeit__erbrechen_konzentrationsschwierigkeiten', 'magenbeschwerden', 'aufstossen']
    O_columns = ['allgemeines_unbehagen_bzw_unwohlsein', 'ermuedung', 'kopfschmerzen', 'ueberanstrengte_augen', 'schwierigkeiten_mit_sehschaerfe', 'uebelkeit__erbrechen_konzentrationsschwierigkeiten', 'verschwommene_sicht']
    D_columns = ['uebelkeit__erbrechen_konzentrationsschwierigkeiten', 'kopfschmerzen', 'druckgefuehl_im_kopfbereich', 'verschwommene_sicht', 'schwindelgefuehl__bei_geoeffneten_augen_', 'schwindelgefuehl__bei_geschlossenen_augen_', 'gleichgewichtsstoerungen']
    pre_ssq_df_tmp['N'] = pre_ssq_df_tmp[N_columns].astype(int).sum(axis=1) * 9.54
    pre_ssq_df_tmp['O'] = pre_ssq_df_tmp[O_columns].astype(int).sum(axis=1) * 7.58
    pre_ssq_df_tmp['D'] = pre_ssq_df_tmp[D_columns].astype(int).sum(axis=1) * 13.92
    pre_ssq_df_tmp['TS'] = (pre_ssq_df_tmp[N_columns].astype(int).sum(axis=1) + pre_ssq_df_tmp[O_columns].astype(int).sum(axis=1) + pre_ssq_df_tmp[D_columns].astype(int).sum(axis=1)) * 3.74
    pre_ssq_df_tmp['test'] = 'before'
    pre_ssq_df = pd.concat([pre_ssq_df, pre_ssq_df_tmp])

    columns_to_keep = ['subject_number', 'test_id', 'radio_es_wuerde_mir_nichts_ausmachen__an_einer_lauten_strasse_zu_wohnen__wenn_meine_wohnung_schoen_waere', 'radio_mir_fallt_laerm_heutzutage_mehr_auf_als_frueher', 'radio_es_sollte_niemanden_gross_stoeren__wenn_ein_anderer_ab_und_zu_seine_stereoanlage_voll_aufdreht', 'radio_im_kino_stoert_mich_fluestern_und_rascheln_von_bonbonpapier', 'radio_ich_werde_leicht_durch_laerm_geweckt', 'radio_wenn_es_an_meinem_arbeitsplatz_iaut_ist__dann_versuche_ich__tuer_oder_fenster_zu_schliessen_oder_anderswo_weiterzuarbeiten', 'radio_es_aergert_mich__wenn_meine_nachbarn_laut_werden', 'radio_an_die_meisten_geraeusche_gewoehne_ich_mich_ohne_grosse_schwierigkeiten', 'radio_es_wuerde_mir_etwas_ausmachen__wenn_eine_wohnung__die_ich_gerne_mieten_wuerde__gegenueber_der_feuerwache_laege', 'radio_manchmal_gehen_mir_geraeusche_auf_die_nerven_und_aergern_mich', 'radio_sogar_musik__die_ich_eigentlich_mag__stoert_mich__wenn_ich_mich_konzentrieren_moechte', 'radio_es_wuerde_mich_nicht_stoeren__die_alltagsgeraeusche_meiner_nachbarn__zb_schritte__wasserrauschen__zu_hoeren', 'radio_wenn_ich_allein_sein_moechte__stoeren_mich_geraeusche_von_ausserhalb', 'radio_ich_kann_mich_gut_konzentrieren__egal_was_um_mich_herum_geschieht', 'radio_in_der_bibliothek_macht_es_mir_nichts_aus__wenn_sich_leute_unterhalten__solange_dies_leise_geschieht', 'radio_oft_wuensche_ich_mir_voellige_stille', 'radio_motorraeder_sollten_besser_schallgedaempft_sein', 'radio_es_faellt_mir_schwer__mich_an_einem_lauten_ort_zu_entspannen', 'radio_ich_werde_wuetend_auf_leute__die_laerm_machen__der_mich_vom_einschlafen_oder_vom_fortkommen_in_der_arbeit_abhaelt', 'radio_es_wuerde_mir_nichts_ausmachen__in_einer_wohnung_mit_duennen_waenden_zu_leben', 'radio_ich_bin_geraeuschempfindlich']
    weinstein_df_tmp = info_df[columns_to_keep]
    weinstein_df_tmp = weinstein_df_tmp.rename(columns=lambda x: x.replace('radio_', ''))
    columns_to_invert = ['es_wuerde_mir_nichts_ausmachen__an_einer_lauten_strasse_zu_wohnen__wenn_meine_wohnung_schoen_waere', 'es_sollte_niemanden_gross_stoeren__wenn_ein_anderer_ab_und_zu_seine_stereoanlage_voll_aufdreht', 'an_die_meisten_geraeusche_gewoehne_ich_mich_ohne_grosse_schwierigkeiten', 'es_wuerde_mich_nicht_stoeren__die_alltagsgeraeusche_meiner_nachbarn__zb_schritte__wasserrauschen__zu_hoeren', 'ich_kann_mich_gut_konzentrieren__egal_was_um_mich_herum_geschieht', 'in_der_bibliothek_macht_es_mir_nichts_aus__wenn_sich_leute_unterhalten__solange_dies_leise_geschieht', 'es_wuerde_mir_nichts_ausmachen__in_einer_wohnung_mit_duennen_waenden_zu_leben']
    for col in columns_to_invert:
        weinstein_df_tmp[col] = weinstein_df_tmp[col].astype(int).apply(lambda x: 4 - x)
    weinstein_df = pd.concat([weinstein_df, weinstein_df_tmp])

d(weinstein_df)
d(pre_ssq_df)

# avrateNG - Post Simulator Sickness Questionnaire (SSQ)
post_ssq_df = pd.DataFrame()

for test_id, test_name in zip(test_ids, test_names):
    df = pd.read_csv("../subjective_data/{}/avrateNG/_post_info.csv".format(test_name), index_col=0)

    info_df = pd.json_normalize(df['info_json'].apply(json.loads))
    info_df['subject_number'] = range(1, len(info_df) + 1)
    info_df['test_id'] = test_id
    columns_to_keep = ['subject_number', 'test_id', 'radio_allgemeines_unbehagen_bzw_unwohlsein', 'radio_ermuedung', 'radio_kopfschmerzen', 'radio_ueberanstrengte_augen', 'radio_schwierigkeiten_mit_sehschaerfe', 'radio_erhoehte_speichelbildung', 'radio_schwitzen', 'radio_uebelkeit__erbrechen_konzentrationsschwierigkeiten', 'radio_druckgefuehl_im_kopfbereich', 'radio_verschwommene_sicht', 'radio_schwindelgefuehl__bei_geoeffneten_augen_', 'radio_schwindelgefuehl__bei_geschlossenen_augen_', 'radio_gleichgewichtsstoerungen', 'radio_magenbeschwerden', 'radio_aufstossen']
    post_ssq_df_tmp = info_df[columns_to_keep]
    post_ssq_df_tmp = post_ssq_df_tmp.rename(columns=lambda x: x.replace('radio_', ''))
    N_columns = ['allgemeines_unbehagen_bzw_unwohlsein', 'erhoehte_speichelbildung', 'schwitzen', 'uebelkeit__erbrechen_konzentrationsschwierigkeiten', 'uebelkeit__erbrechen_konzentrationsschwierigkeiten', 'magenbeschwerden', 'aufstossen']
    O_columns = ['allgemeines_unbehagen_bzw_unwohlsein', 'ermuedung', 'kopfschmerzen', 'ueberanstrengte_augen', 'schwierigkeiten_mit_sehschaerfe', 'uebelkeit__erbrechen_konzentrationsschwierigkeiten', 'verschwommene_sicht']
    D_columns = ['uebelkeit__erbrechen_konzentrationsschwierigkeiten', 'kopfschmerzen', 'druckgefuehl_im_kopfbereich', 'verschwommene_sicht', 'schwindelgefuehl__bei_geoeffneten_augen_', 'schwindelgefuehl__bei_geschlossenen_augen_', 'gleichgewichtsstoerungen']
    post_ssq_df_tmp['N'] = post_ssq_df_tmp[N_columns].astype(int).sum(axis=1) * 9.54
    post_ssq_df_tmp['O'] = post_ssq_df_tmp[O_columns].astype(int).sum(axis=1) * 7.58
    post_ssq_df_tmp['D'] = post_ssq_df_tmp[D_columns].astype(int).sum(axis=1) * 13.92
    post_ssq_df_tmp['TS'] = (post_ssq_df_tmp[N_columns].astype(int).sum(axis=1) + post_ssq_df_tmp[O_columns].astype(int).sum(axis=1) + post_ssq_df_tmp[D_columns].astype(int).sum(axis=1)) * 3.74
    post_ssq_df_tmp['test'] = 'after'
    post_ssq_df = pd.concat([post_ssq_df, post_ssq_df_tmp])

d(post_ssq_df)

# avrageNG - IPQ
ipq_df = pd.DataFrame()

for test_id, test_name in zip(test_ids, test_names):
    df = pd.read_csv("../subjective_data/{}/avrateNG/_post_info.csv".format(test_name), index_col=0)

    info_df = pd.json_normalize(df['info_json'].apply(json.loads))
    info_df['subject_number'] = range(1, len(info_df) + 1)
    info_df['test_id'] = test_id
    columns_to_keep = ["subject_number",
                       "test_id", 
                       "radio_in_der_computererzeugten_welt_hatte_ich_den_eindruck__dort_gewesen_zu_sein",
                       "radio_ich_hatte_das_gefuehl__dass_die_virtuelle_umgebung_hinter_mir_weitergeht",
                       "radio_ich_hatte_das_gefuehl__nur_bilder_zu_sehen",
                       "radio_ich_hatte_nicht_das_gefuehl__in_dem_virtuellen_raum_zu_sein",
                       "radio_ich_hatte_das_gefuehl__in_dem_virtuellen_raum_zu_handeln_statt_etwas_von_aussen_zu_bedienen",
                       "radio_ich_fuehlte_mich_im_virtuellen_raum_anwesend",
                       "radio_wie_bewusst_war_ihnen_die_reale_welt__waehrend_sie_sich_durch_die_virtuelle_welt_bewegten__zb_geraeusche__raumtemperatur__andere_personen_etc_",
                       "radio_meine_reale_umgebung_war_mir_nicht_mehr_bewusst",
                       "radio_ich_achtete_noch_auf_die_reale_umgebung",
                       "radio_meine_aufmerksamkeit_war_von_der_virtuellen_welt_voellig_in_bann_gezogen",
                       "radio_wie_real_erschien_ihnen_die_virtuelle_umgebung",
                       "radio_wie_sehr_glich_ihr_erleben_der_virtuellen_umgebung_dem_erleben_einer_realen_umgebung",
                       "radio_wie_real_erschien_ihnen_die_virtuelle_welt",
                       "radio_die_virtuelle_welt_erschien_mir_wirklicher_als_die_reale_welt"]
    ipq_df_tmp = info_df[columns_to_keep]
    ipq_df_tmp.rename(columns={'radio_in_der_computererzeugten_welt_hatte_ich_den_eindruck__dort_gewesen_zu_sein':'G1'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_ich_hatte_das_gefuehl__dass_die_virtuelle_umgebung_hinter_mir_weitergeht':'SP1'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_ich_hatte_das_gefuehl__nur_bilder_zu_sehen':'SP2'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_ich_hatte_nicht_das_gefuehl__in_dem_virtuellen_raum_zu_sein':'SP3'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_ich_hatte_das_gefuehl__in_dem_virtuellen_raum_zu_handeln_statt_etwas_von_aussen_zu_bedienen':'SP4'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_ich_fuehlte_mich_im_virtuellen_raum_anwesend':'SP5'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_wie_bewusst_war_ihnen_die_reale_welt__waehrend_sie_sich_durch_die_virtuelle_welt_bewegten__zb_geraeusche__raumtemperatur__andere_personen_etc_':'INV1'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_meine_reale_umgebung_war_mir_nicht_mehr_bewusst':'INV2'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_ich_achtete_noch_auf_die_reale_umgebung':'INV3'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_meine_aufmerksamkeit_war_von_der_virtuellen_welt_voellig_in_bann_gezogen':'INV4'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_wie_real_erschien_ihnen_die_virtuelle_umgebung':'REAL1'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_wie_sehr_glich_ihr_erleben_der_virtuellen_umgebung_dem_erleben_einer_realen_umgebung':'REAL2'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_wie_real_erschien_ihnen_die_virtuelle_welt':'REAL3'}, inplace=True)
    ipq_df_tmp.rename(columns={'radio_die_virtuelle_welt_erschien_mir_wirklicher_als_die_reale_welt':'REAL4'}, inplace=True)
    ipq_df_tmp = ipq_df_tmp.apply(pd.to_numeric)
    ipq_df_tmp["SP2U"] = -1 * ipq_df_tmp["SP2"] + 6
    ipq_df_tmp["INV3U"] = -1 * ipq_df_tmp["INV3"] + 6
    ipq_df_tmp["REAL1U"] = -1 * ipq_df_tmp["REAL1"] + 6
    ipq_df_tmp["SP"] = ipq_df_tmp[["SP1", "SP2U", "SP3", "SP4", "SP5"]].mean(axis=1)
    ipq_df_tmp["INV"] = ipq_df_tmp[["INV1", "INV2", "INV3U", "INV4"]].mean(axis=1)
    ipq_df_tmp["REAL"] = ipq_df_tmp[["REAL1U", "REAL2", "REAL3", "REAL4"]].mean(axis=1)
    ipq_df = pd.concat([ipq_df, ipq_df_tmp])
d(ipq_df)

# avrateNG - 5 questions from NASA TLX & 1 question from Schmutz et al.
nasatlx_df = pd.DataFrame()

for test_id, test_name in zip(test_ids, test_names):
    df = pd.read_csv('../subjective_data/{}/avrateNG/_ratings.csv'.format(test_name), index_col=False)
    
    df_filtered = df[df['rating_type'].str.startswith('range_')].copy()
    df_filtered['test_id'] = test_id
    
    unique_user_ids = df_filtered['user_ID'].unique()
    new_user_ids = {old_id: new_id+1 for new_id, old_id in enumerate(unique_user_ids)}
    df_filtered.loc[:, 'subject_number'] = df_filtered['user_ID'].map(new_user_ids)
    
    unique_stimuli_files = df_filtered['stimuli_file'].unique()
    new_stimuli_ids = {old_file: new_id+5 for new_id, old_file in enumerate(unique_stimuli_files)}
    df_filtered.loc[:, 'scene'] = df_filtered['stimuli_file'].map(new_stimuli_ids)

    rating_type_mapping = {
        "range_wie_anspruchsvoll_war_die_durchfuehrung_der_aufgabe_fuer_sie": "nasatlx_mental_demand",
        "range_wie_viel_zeitdruck_hatten_sie_bei_der_durchfuehrung_der_aufgabe": "nasatlx_temporal_demand",
        "range_wie_hoch_war_ihr_erfolgserlebnis_bei_der_durchfuehrung_der_aufgabe": "nasatlx_performance",
        "range_wie_hart_mussten_sie_arbeiten__um_ihr_leistungsniveau_zu_erreichen": "nasatlx_effort",
        "range_wie_unsicher__entmutigt__gestresst_und_genervt_haben_sie_sich_waehrend_der_aufgabe_gefuehlt": "nasatlx_frustration",
        "range_wie_hoch_war_die_von_ihnen_empfundene_anstrengung_beim_zuhoeren": "listening_effort"
    }
    df_filtered.loc[:, 'rating_type'] = df_filtered['rating_type'].map(rating_type_mapping)

    df_filtered = df_filtered[['subject_number', 'test_id', 'scene', 'rating_type', 'rating']]
    
    df_pivoted = df_filtered.pivot(index=['subject_number', 'test_id', 'scene'], columns='rating_type', values='rating').reset_index()
    df_pivoted = df_pivoted.apply(pd.to_numeric)
    df_pivoted['nasatlx_performance'] = 100 - df_pivoted['nasatlx_performance']  # Invert performance column because of question style
    df_pivoted['nasatlx_mental_workload_score'] = df_pivoted[['nasatlx_mental_demand', 'nasatlx_temporal_demand', 'nasatlx_performance', 'nasatlx_effort', 'nasatlx_frustration']].mean(axis=1)

    df_melted = pd.melt(df_pivoted, id_vars=['subject_number', 'test_id', 'scene'], 
                        value_vars=['nasatlx_effort', 'nasatlx_frustration', 'nasatlx_mental_demand', 'nasatlx_performance', 'nasatlx_temporal_demand', 'nasatlx_mental_workload_score', 'listening_effort'],
                        var_name='rating_type', value_name='rating')

    nasatlx_df = pd.concat([nasatlx_df, df_melted])
d(nasatlx_df)

# Unity - behaviour CSV (timestamp, controller interaction, head rotation, etc.)
headrotation_df = pd.DataFrame()

for test_id, test_name in zip(test_ids, test_names):
    for file in glob.glob('../subjective_data/{}/Unity/*.csv'.format(test_name)):
        if not "training" in file:
            subject = Path(file).stem.split('_')[0]
            total_stories = Path(file).stem.split('_')[1]
            df = pd.read_csv(file, index_col=False)

            # Compute total time needed
            total_time_s = (pd.to_datetime(df['TimeStamp'].iloc[-1], format='%d.%m.%Y %H:%M:%S') - pd.to_datetime(df['TimeStamp'].iloc[0], format='%d.%m.%Y %H:%M:%S')).total_seconds()
            if total_time_s > 120:  # for some reason, some are 121s
                total_time_s = 120

            # Compute degrees explored
            pitch_explored = df['VRCameraPitch'].diff().abs().sum()
            def adjusted_diff(yaw_series):
                diff = yaw_series.diff().dropna()
                adjusted = diff.apply(lambda x: x if abs(x) <= 180 else (x - 360 if x > 180 else x + 360))
                return adjusted.abs().sum()
            yaw_explored = adjusted_diff(df['VRCameraYaw'])
            degrees_explored = pitch_explored + yaw_explored
            # minyaw = abs(df['VRCameraYaw'].min())
            # maxyaw = abs(df['VRCameraYaw'].max())
            # d(f'Subject: {subject}, Test: {test_id}, Stories: {total_stories}, MinYaw: {minyaw}, MaxYaw: {maxyaw}')

            # Compute delta(max-min)
            delta_minmax_pitch = abs(df['VRCameraPitch'].min()) + abs(df['VRCameraPitch'].max())
            delta_minmax_yaw = abs(df['VRCameraYaw'].min()) + abs(df['VRCameraYaw'].max())
            if delta_minmax_yaw > 360:  # Somehow this sometimes happens, as positive yaw values can be up to 184.99999 degrees for whatever f****** reason
                delta_minmax_yaw = 360

            # Compute number of direction changes for yaw
            def compute_number_yaw_direction_changes(yaw_series, test_id):
                test_id = int(test_id)
                # if test_id in [1, 2]:
                #     yaw_series = yaw_series.iloc[::2]
                diff = yaw_series.diff().dropna()
                if test_id in [1, 2]:  # to compensate for recording at 90 Hz for tests 1 and 2 and 45 Hz for test 3
                    threshold = 0.1
                    signs = diff.apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))
                else:
                    threshold = 0.2
                    signs = diff.apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))
                count = 0
                prev_sign = 0
                for current_sign in signs:
                    if current_sign != 0 and current_sign != prev_sign:
                        count += 1
                        prev_sign = current_sign
                return count
            number_yaw_direction_changes = compute_number_yaw_direction_changes(df['VRCameraYaw'], test_id)
            # Compute number of direction changes per second
            number_yaw_direction_changes_per_s = number_yaw_direction_changes / total_time_s

            tmp_dict = {
                'subject_number': [subject],
                'test_id': [test_id],
                'total_stories_count': [total_stories],
                'total_time_s': [total_time_s],
                'total_pitch_explored': [pitch_explored],
                'total_yaw_explored': [yaw_explored],
                'total_degrees_explored': [degrees_explored],
                'delta_minmax_pitch': delta_minmax_pitch,
                'delta_minmax_yaw': delta_minmax_yaw,
                'number_yaw_direction_changes': number_yaw_direction_changes,
                'number_yaw_direction_changes_per_s': number_yaw_direction_changes_per_s
            }
            tmp_df = pd.DataFrame(tmp_dict)
            headrotation_df = pd.concat([headrotation_df, tmp_df], ignore_index=True)
headrotation_df = headrotation_df.apply(pd.to_numeric)
d(headrotation_df)


# # SSQ

# In[170]:


deep_palette = sns.color_palette("deep", 3)

def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = mcolors.to_rgb(color)
    white = mcolors.to_rgb('white')
    lightened_color = mcolors.to_rgb(tuple([(1 - amount) * c[i] + amount * white[i] for i in range(3)]))
    return lightened_color

paired_palette = []
for color in deep_palette:
    before_color = lighten_color(color, 0.4)
    after_color = lighten_color(color, 0)
    paired_palette.append(before_color)
    paired_palette.append(after_color)

ssq_combined_df = pd.concat([pre_ssq_df, post_ssq_df])
ssq_combined_df["test_combined"] = ssq_combined_df["test_id"] + "_" + ssq_combined_df["test"]

melted_df = pd.melt(ssq_combined_df, id_vars='test_combined', value_vars=['N', 'O', 'D', 'TS'], var_name='category', value_name='mean_ssq_value')
hue_order = ['1_before', '1_after', '2_before', '2_after', '3_before', '3_after']
plt.figure(figsize=(6.6, 5))
ax = sns.barplot(x='category', y='mean_ssq_value', hue='test_combined', palette=paired_palette, data=melted_df, hue_order=hue_order, errorbar=('ci', 95), capsize=.4)
ax.set_xlabel("Simulator Sickness Questionnaire score")
ax.set_xticklabels(["Nausea (N)", "Oculomotor (O)", "Disorientation (D)", "Total score (TS)"])
ax.set_ylabel("Simulator Sickness Questionnaire score value")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='Experimental condition and time of measurement', labels=['360° diotic_before', '360° diotic_after', '360° binaural_before', '360° binaural_after', 'CGI binaural_before', 'CGI binaural_after'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

# ax = sns.barplot(x='category', y='mean_ssq_value', hue='test_combined', data=pd.melt(ssq_combined_df, id_vars='test_combined', value_vars=['N', 'O', 'D', 'TS'], var_name='category', value_name='mean_ssq_value'), errorbar=('ci', 95))
d(ax)
save_fig(ax, "", "plots/mean_ssq_before_after_test.pdf")
save_fig(ax, "", "plots/mean_ssq_before_after_test.png")


# In[171]:


ssq_combined_df_aov_mod = ssq_combined_df
subject_numbers = (
    list(np.tile(np.arange(1, 26), 2)) +
    list(np.tile(np.arange(26, 62), 2)) +
    list(np.tile(np.arange(63, 97), 2))
)
ssq_combined_df_aov_mod['subject_number'] = subject_numbers
d(ssq_combined_df_aov_mod)

ssq_combined_df_aov_mod_reduced = ssq_combined_df_aov_mod[["test_id", "test", "test_combined", "subject_number", "N", "O", "D", "TS"]]
ssq_combined_df_aov_mod_reduced = ssq_combined_df_aov_mod_reduced.reset_index(drop=True)
d(ssq_combined_df_aov_mod_reduced)

d("Nausea (N) ART:")
pandas2ri.activate()
r_df = pandas2ri.py2rpy(ssq_combined_df_aov_mod_reduced)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$test <- factor(df$test)
library(ARTool)
m <- art(N ~ test_id * test + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
post_hoc_res = ro.r('''
res <- art.con(m, "test_id:test", adjust="none")
df_res <- as.data.frame(res)
df_filtered <- subset(df_res, contrast %in% c("1,after - 1,before", "2,after - 2,before", "3,after - 3,before"))
df_filtered$p.value_corr <- pmin(df_filtered$p.value * 3, 1)
df_filtered
''')
d(pandas2ri.rpy2py(post_hoc_res))

d("Nausea (N) Old:")
aov = pg.rm_anova(data=ssq_combined_df_aov_mod_reduced, dv="N", within="test", subject="subject_number", correction='auto', effsize="np2")
d(aov)
posthoc = pg.pairwise_tukey(data=ssq_combined_df_aov_mod_reduced, dv="N", between="test_combined")
d(posthoc)



d("Oculumotor (O) ART:")
pandas2ri.activate()
r_df = pandas2ri.py2rpy(ssq_combined_df_aov_mod_reduced)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
ro.r('''
df$test_id <- factor(df$test_id)
df$test <- factor(df$test)
library(ARTool)
m <- art(O ~ test_id * test + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
post_hoc_res = ro.r('''
res <- art.con(m, "test_id:test", adjust="none")
df_res <- as.data.frame(res)
df_filtered <- subset(df_res, contrast %in% c("1,after - 1,before", "2,after - 2,before", "3,after - 3,before"))
df_filtered$p.value_corr <- pmin(df_filtered$p.value * 3, 1)
df_filtered
''')
d(pandas2ri.rpy2py(post_hoc_res))

d("Oculumotor (O) Old:")
aov = pg.rm_anova(data=ssq_combined_df_aov_mod_reduced, dv="O", within="test", subject="subject_number", correction='auto', detailed=True, effsize="np2")
d(aov)
posthoc = pg.pairwise_tukey(data=ssq_combined_df_aov_mod_reduced, dv="O", between="test_combined")
d(posthoc)


d("Disorientation (D) ART:")
pandas2ri.activate()
r_df = pandas2ri.py2rpy(ssq_combined_df_aov_mod_reduced)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
ro.r('''
df$test_id <- factor(df$test_id)
df$test <- factor(df$test)
library(ARTool)
m <- art(D ~ test_id * test + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
post_hoc_res = ro.r('''
res <- art.con(m, "test_id:test", adjust="none")
df_res <- as.data.frame(res)
df_filtered <- subset(df_res, contrast %in% c("1,after - 1,before", "2,after - 2,before", "3,after - 3,before"))
df_filtered$p.value_corr <- pmin(df_filtered$p.value * 3, 1)
df_filtered
''')
d(pandas2ri.rpy2py(post_hoc_res))

d("Disorientation (D) Old:")
aov = pg.rm_anova(data=ssq_combined_df_aov_mod_reduced, dv="D", within="test", subject="subject_number", correction='auto', detailed=True, effsize="np2")
d(aov)
posthoc = pg.pairwise_tukey(data=ssq_combined_df_aov_mod_reduced, dv="D", between="test_combined")
d(posthoc)



d("Total Score (TS) ART:")
pandas2ri.activate()
r_df = pandas2ri.py2rpy(ssq_combined_df_aov_mod_reduced)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
ro.r('''
df$test_id <- factor(df$test_id)
df$test <- factor(df$test)
library(ARTool)
m <- art(TS ~ test_id * test + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
post_hoc_res = ro.r('''
res <- art.con(m, "test_id:test", adjust="none")
df_res <- as.data.frame(res)
df_filtered <- subset(df_res, contrast %in% c("1,after - 1,before", "2,after - 2,before", "3,after - 3,before"))
df_filtered$p.value_corr <- pmin(df_filtered$p.value * 3, 1)
df_filtered
''')
d(pandas2ri.rpy2py(post_hoc_res))

d("Total Score (TS) Old:")
aov = pg.rm_anova(data=ssq_combined_df_aov_mod_reduced, dv="TS", within="test", subject="subject_number", correction='auto', detailed=True, effsize="np2")
d(aov)
posthoc = pg.pairwise_tukey(data=ssq_combined_df_aov_mod_reduced, dv="TS", between="test_combined")
d(posthoc)


# In[172]:


ssq_combined_df_aov_mod2 = ssq_combined_df
ssq_combined_df_aov_mod_before = ssq_combined_df_aov_mod2.loc[ssq_combined_df_aov_mod["test"] == "before"]
ssq_combined_df_aov_mod_before['subject_number'] = list(np.arange(1, 96))
d(ssq_combined_df_aov_mod_before)
ssq_combined_df_aov_mod_after = ssq_combined_df_aov_mod2.loc[ssq_combined_df_aov_mod["test"] == "after"]
ssq_combined_df_aov_mod_after['subject_number'] = list(np.arange(1, 96))
d(ssq_combined_df_aov_mod_after)

shapiro = pg.normality(ssq_combined_df_aov_mod_before, dv="TS", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(ssq_combined_df_aov_mod_before, dv="TS", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.anova(data=ssq_combined_df_aov_mod_before, dv="N", between="test_id", detailed=True)
d(aov)
aov.to_csv("data_eval/anova_ssq_N_before.csv")

aov = pg.anova(data=ssq_combined_df_aov_mod_before, dv="O", between="test_id", detailed=True)
d(aov)
aov.to_csv("data_eval/anova_ssq_O_before.csv")

posthoc = pg.pairwise_tukey(data=ssq_combined_df_aov_mod_before, dv="O", between="test_id")
d(posthoc)
posthoc.to_csv("data_eval/posthoc_ssq_O_before_per_test.csv")

aov = pg.anova(data=ssq_combined_df_aov_mod_before, dv="D", between="test_id", detailed=True)
d(aov)
aov.to_csv("data_eval/anova_ssq_D_before.csv")

aov = pg.anova(data=ssq_combined_df_aov_mod_before, dv="TS", between="test_id", detailed=True)
d(aov)
aov.to_csv("data_eval/anova_ssq_TS_before.csv")


shapiro = pg.normality(ssq_combined_df_aov_mod_after, dv="TS", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(ssq_combined_df_aov_mod_after, dv="TS", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.anova(data=ssq_combined_df_aov_mod_after, dv="TS", between="test_id", detailed=True)
d(aov)
aov.to_csv("data_eval/anova_ssq_TS_after.csv")


# # Weinstein score

# In[173]:


# Plot Weinstein Noise Sensitivity Scale (WNSS)
palette = sns.color_palette("deep")
sns.set_palette(palette)

columns_for_mean = ['es_wuerde_mir_nichts_ausmachen__an_einer_lauten_strasse_zu_wohnen__wenn_meine_wohnung_schoen_waere', 'mir_fallt_laerm_heutzutage_mehr_auf_als_frueher', 'es_sollte_niemanden_gross_stoeren__wenn_ein_anderer_ab_und_zu_seine_stereoanlage_voll_aufdreht', 'im_kino_stoert_mich_fluestern_und_rascheln_von_bonbonpapier', 'ich_werde_leicht_durch_laerm_geweckt', 'wenn_es_an_meinem_arbeitsplatz_iaut_ist__dann_versuche_ich__tuer_oder_fenster_zu_schliessen_oder_anderswo_weiterzuarbeiten', 'es_aergert_mich__wenn_meine_nachbarn_laut_werden', 'an_die_meisten_geraeusche_gewoehne_ich_mich_ohne_grosse_schwierigkeiten', 'es_wuerde_mir_etwas_ausmachen__wenn_eine_wohnung__die_ich_gerne_mieten_wuerde__gegenueber_der_feuerwache_laege', 'manchmal_gehen_mir_geraeusche_auf_die_nerven_und_aergern_mich', 'sogar_musik__die_ich_eigentlich_mag__stoert_mich__wenn_ich_mich_konzentrieren_moechte', 'es_wuerde_mich_nicht_stoeren__die_alltagsgeraeusche_meiner_nachbarn__zb_schritte__wasserrauschen__zu_hoeren', 'wenn_ich_allein_sein_moechte__stoeren_mich_geraeusche_von_ausserhalb', 'ich_kann_mich_gut_konzentrieren__egal_was_um_mich_herum_geschieht', 'in_der_bibliothek_macht_es_mir_nichts_aus__wenn_sich_leute_unterhalten__solange_dies_leise_geschieht', 'oft_wuensche_ich_mir_voellige_stille', 'motorraeder_sollten_besser_schallgedaempft_sein', 'es_faellt_mir_schwer__mich_an_einem_lauten_ort_zu_entspannen', 'ich_werde_wuetend_auf_leute__die_laerm_machen__der_mich_vom_einschlafen_oder_vom_fortkommen_in_der_arbeit_abhaelt', 'es_wuerde_mir_nichts_ausmachen__in_einer_wohnung_mit_duennen_waenden_zu_leben', 'ich_bin_geraeuschempfindlich']
weinstein_df = weinstein_df.apply(pd.to_numeric)
weinstein_df['mean_weinstein_score'] = weinstein_df[columns_for_mean].mean(axis=1)
d(weinstein_df)
# std_error = weinstein_df[columns_for_mean].std(axis=1) / np.sqrt(len(columns_for_mean))
ax = sns.barplot(x='test_id', y='mean_weinstein_score', hue="test_id", legend=False, palette="deep", data=weinstein_df)
ax.set_xlabel("Experimental condition")
ax.set_xticklabels(['360° diotic', '360° binaural', 'CGI binaural'])
ax.set_ylabel("Weinstein score")
# mean_weinstein_all_subjects = weinstein_df['mean_weinstein_score'].mean()
# plt.axhline(y=mean_weinstein_all_subjects, color='r', linestyle='--', label=f'Mean: {mean_percentage:.2f}%', linewidth=2)
d(ax)
save_fig(ax, "", "plots/barplot_weinstein_score_per_test.pdf")
save_fig(ax, "", "plots/barplot_weinstein_score_per_test.png")


# In[174]:


ax = sns.boxplot(x='test_id', y='mean_weinstein_score', hue="test_id", legend=False, palette="deep", data=weinstein_df)
ax.set_xlabel("Experimental condition")
ax.set_xticklabels(['360° diotic', '360° binaural', 'CGI binaural'])
ax.set_ylabel("Weinstein score")
d(ax)
save_fig(ax, "", "plots/boxplot_weinstein_score_per_test.pdf")
save_fig(ax, "", "plots/boxplot_weinstein_score_per_test.png")


# In[175]:


weinstein_min = weinstein_df.groupby("test_id").mean_weinstein_score.min().sort_values(ascending=False)
weinstein_max = weinstein_df.groupby("test_id").mean_weinstein_score.max().sort_values(ascending=False)
weinstein_mean = weinstein_df.groupby("test_id").mean_weinstein_score.mean().sort_values(ascending=False)
weinstein_median = weinstein_df.groupby("test_id").mean_weinstein_score.median().sort_values(ascending=False)
weinstein_std = weinstein_df.groupby("test_id").mean_weinstein_score.std().sort_values(ascending=False)
weinstein_eval = pd.DataFrame({
    'Min': weinstein_min,
    'Max': weinstein_max,
    'Mean': weinstein_mean,
    'Median': weinstein_median,
    'SD': weinstein_std
})
weinstein_eval = weinstein_eval.T.applymap(lambda x: f"{x:.3f}")
d(weinstein_eval)
d(weinstein_eval.style.to_latex())


# In[176]:


weinstein_df_aov_mod = weinstein_df
weinstein_df_aov_mod['subject_number'] = list(np.arange(1, 96))
d(weinstein_df_aov_mod)

shapiro = pg.normality(weinstein_df_aov_mod, dv="mean_weinstein_score", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(weinstein_df_aov_mod, dv="mean_weinstein_score", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.anova(data=weinstein_df_aov_mod, dv="mean_weinstein_score", between="test_id", detailed=True)
d(aov)
aov.to_csv("data_eval/anova_weinstein.csv")


# # IPQ

# In[177]:


ipq_df_melted = ipq_df.melt(id_vars=["subject_number", "test_id"], value_vars=["G1", "SP", "INV", "REAL"],
                    var_name="rating_type", value_name="rating")
# G1 = general presence
# SP = spatial presence
# INV = involvement
# REAL = experienced realism
d(ipq_df_melted)
plt.figure(figsize=(5.5, 4))
ax = sns.boxplot(x='test_id', y='rating', hue="rating_type", palette="husl", data=ipq_df_melted)
ax.set_xlabel("Experimental condition")
ax.set_xticklabels(['360° diotic', '360° binaural', 'CGI binaural'])
ax.set_ylabel("IPQ value")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='IPQ rating', labels=["General presence (G1)", "Spatial presence (SP)", "Involvement (INV)", "Experienced realism (REAL)"], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

d(ax)
save_fig(ax, "", "plots/boxplot_ipq_evaluation.pdf")
save_fig(ax, "", "plots/boxplot_ipq_evaluation.png")
medians = ipq_df_melted.groupby(['test_id', 'rating_type'])['rating'].median().reset_index()
d("Medians:")
d(medians)


# In[178]:


plt.figure(figsize=(5.5, 4.5))
ax = sns.barplot(x='test_id', y='rating', hue="rating_type", palette="husl", data=ipq_df_melted, errorbar=('ci', 95), capsize=.5)
ax.set_xlabel("Experimental condition")
ax.set_xticklabels(['360° diotic', '360° binaural', 'CGI binaural'])
ax.set_ylabel("Igroup Presence Questionnaire value")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='Igroup Presence Questionnaire dimension', labels=["General presence (G1)", "Spatial presence (SP)", "Involvement (INV)", "Experienced realism (REAL)"], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

d(ax)
save_fig(ax, "", "plots/barplot_ipq_evaluation.pdf")
save_fig(ax, "", "plots/barplot_ipq_evaluation.png")
d(ipq_df_melted.groupby(['test_id', 'rating_type'])['rating'].mean().reset_index())


# In[179]:


ipq_df_aov_mod = ipq_df
ipq_df_aov_mod['subject_number'] = list(np.arange(1, 96))
d(ipq_df_aov_mod)


shapiro = pg.normality(ipq_df_aov_mod, dv="G1", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(ipq_df_aov_mod, dv="G1", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.anova(data=ipq_df_aov_mod, dv="G1", between="test_id", detailed=True)
d(aov)
aov.to_csv("data_eval/anova_ipq_G1.csv")
posthoc = pg.pairwise_tukey(data=ipq_df_aov_mod, dv="G1", between="test_id")
d(posthoc)


shapiro = pg.normality(ipq_df_aov_mod, dv="SP", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(ipq_df_aov_mod, dv="SP", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.anova(data=ipq_df_aov_mod, dv="SP", between="test_id", detailed=True)
d(aov)
aov.to_csv("data_eval/anova_ipq_SP.csv")
posthoc = pg.pairwise_tukey(data=ipq_df_aov_mod, dv="SP", between="test_id")
d(posthoc)


shapiro = pg.normality(ipq_df_aov_mod, dv="INV", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(ipq_df_aov_mod, dv="INV", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.anova(data=ipq_df_aov_mod, dv="INV", between="test_id", detailed=True)
d(aov)
aov.to_csv("data_eval/anova_ipq_INV.csv")
posthoc = pg.pairwise_tukey(data=ipq_df_aov_mod, dv="INV", between="test_id")
d(posthoc)


shapiro = pg.normality(ipq_df_aov_mod, dv="REAL", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(ipq_df_aov_mod, dv="REAL", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.anova(data=ipq_df_aov_mod, dv="REAL", between="test_id", detailed=True)
d(aov)
aov.to_csv("data_eval/anova_ipq_REAL.csv")
posthoc = pg.pairwise_tukey(data=ipq_df_aov_mod, dv="REAL", between="test_id")
d(posthoc)
aov.to_csv("data_eval/posthoc_ipq_REAL.csv")


# In[180]:


d("General presence (G1)")
pandas2ri.activate()
r_df = pandas2ri.py2rpy(ipq_df_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$test <- factor(df$test)
library(ARTool)
m <- art(G1 ~ test_id, data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="bonferroni")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))


d("Spatial presence (SP)")
pandas2ri.activate()
r_df = pandas2ri.py2rpy(ipq_df_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$test <- factor(df$test)
library(ARTool)
m <- art(SP ~ test_id, data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="bonferroni")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))


d("Involvement (INV)")
pandas2ri.activate()
r_df = pandas2ri.py2rpy(ipq_df_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$test <- factor(df$test)
library(ARTool)
m <- art(INV ~ test_id, data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="bonferroni")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))


d("Experienced realism (REAL)")
pandas2ri.activate()
r_df = pandas2ri.py2rpy(ipq_df_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$test <- factor(df$test)
library(ARTool)
m <- art(REAL ~ test_id, data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="bonferroni")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))


# # Further data processing (ETL)

# In[181]:


# Unity speaker_story_mapping
# Start to read every .json file, start with Scene5 as Scene1-4 is training
# And do it for every subject participating in the test
# Also load the _ratings.csv file from AVrate to get the corresponding ratings from
# the two questionnaires: NASA TLX & Schmutz et al.
scenes_to_read = ['5', '6', '7', '8', '9', '10', '11', '12', '13']
subjects_to_read = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
normalized_dfs = []
for test_id, test_name in zip(test_ids, test_names):
    for subject_to_read in subjects_to_read:
        for scene_to_read in scenes_to_read:
            try:
                df = pd.read_json('../subjective_data/{}/Unity/ecoclass-vr_chairs_speaker_story_mapping_{}_Scene{}.json'.format(test_name, subject_to_read, scene_to_read))
                scene_number = int(re.findall(r'\d+', scene_to_read)[0])
                nested_data = df.at[scene_number - 1, 'chairs_speaker_story_mapping']
                normalized_df = pd.json_normalize(nested_data)
                normalized_df["test_id"] = test_id
                normalized_df["scene"] = scene_to_read
                normalized_df["subject_number"] = subject_to_read
                normalized_dfs.append(normalized_df)
            except:
                None

speaker_story_mapping_df = pd.concat(normalized_dfs, ignore_index=True)
speaker_story_mapping_df = speaker_story_mapping_df.apply(pd.to_numeric)

# Get the amount of total stories, the amount of correctly assigned stories and the percentage of correctly assigned stories
story_columns = [col for col in speaker_story_mapping_df.columns if '.story' in col]
assigned_story_columns = [col for col in speaker_story_mapping_df.columns if '.assigned_story' in col]
story_columns.sort()
assigned_story_columns.sort()
speaker_story_mapping_df['total_stories_count'] = speaker_story_mapping_df[story_columns].apply(lambda x: (x != 11).sum(), axis=1)
speaker_story_mapping_df['assigned_stories_count'] = speaker_story_mapping_df[assigned_story_columns].apply(lambda x: (x != 11).sum(), axis=1)
speaker_story_mapping_df['correctly_assigned_stories_count'] = 0
for story_col, assigned_story_col in zip(story_columns, assigned_story_columns):
    speaker_story_mapping_df['correctly_assigned_stories_count'] += (speaker_story_mapping_df[story_col] != 11) & (speaker_story_mapping_df[story_col] == speaker_story_mapping_df[assigned_story_col]).astype(int)
speaker_story_mapping_df['correctly_assigned_stories_percentage'] = speaker_story_mapping_df['correctly_assigned_stories_count'] / speaker_story_mapping_df['total_stories_count'] * 100

# Calculate wrongly_assigned_stories
speaker_story_mapping_df['wrongly_assigned_stories_count'] = 0
speaker_story_mapping_df['no_assigned_stories_count'] = 0
for i in range(1, 21):
    story_col = f'chair_{i}.story'
    assigned_story_col = f'chair_{i}.assigned_story'
    speaker_story_mapping_df['wrongly_assigned_stories_count'] += ((speaker_story_mapping_df[story_col] != 11) & (speaker_story_mapping_df[assigned_story_col] != 11) & (speaker_story_mapping_df[story_col] != speaker_story_mapping_df[assigned_story_col])).astype(int)
    speaker_story_mapping_df['no_assigned_stories_count'] += ((speaker_story_mapping_df[story_col] != 11) & (speaker_story_mapping_df[assigned_story_col] == 11)).astype(int)
speaker_story_mapping_df['wrongly_assigned_stories_percentage'] = (speaker_story_mapping_df['wrongly_assigned_stories_count'] / speaker_story_mapping_df['total_stories_count']) * 100
speaker_story_mapping_df['no_assigned_stories_percentage'] = (speaker_story_mapping_df['no_assigned_stories_count'] / speaker_story_mapping_df['total_stories_count']) * 100

number_of_story_in_test = [(i % 9) + 1 for i in range(len(speaker_story_mapping_df))]
speaker_story_mapping_df['number_of_trial_in_test'] = number_of_story_in_test

d(speaker_story_mapping_df)
# speaker_story_mapping_df.to_excel("speaker_story_mapping_df.xlsx", sheet_name='speaker_story_mapping_df')


# # Speaker-story-mapping performance (percentage of correctly assigned stories)

# In[182]:


ax = sns.lineplot(x='total_stories_count', y='correctly_assigned_stories_percentage', hue='test_id', data=speaker_story_mapping_df, palette="deep")
ax.set_xlabel("Total number of stories")
ax.set_ylabel("Correctly assigned stories [%]")

# Add the results from Ahrens et al. (2019)
ahrens_x = [2, 3, 4, 5, 6, 7, 8, 9, 10]
ahrens_y = [100, 100, 95, 76, 82, 50, 22, 3, 0]
ax.plot(ahrens_x, ahrens_y, color='gray', linestyle='-', linewidth=1)
ahrens_handle = Line2D([0], [0], color='gray', linestyle='-', linewidth=1)
handles, labels = ax.get_legend_handles_labels()
handles.append(ahrens_handle)
labels.append('Ahrens et al. (2019)')

ax.legend(handles=handles, title='Experimental condition', labels=['360° diotic', '360° binaural', 'CGI binaural', 'Ahrens et al. (2019)'])

d(ax)
save_fig(ax, "Correctly assigned stories per test", "plots/lineplot_speaker_story_mapping_percentage_per_test.pdf")
save_fig(ax, "Correctly assigned stories per test", "plots/lineplot_speaker_story_mapping_percentage_per_test.png")
d(speaker_story_mapping_df.groupby(["test_id", "total_stories_count"])["correctly_assigned_stories_percentage"].mean().reset_index())


# In[183]:


# 1) Put Ahrens data in a DataFrame
ahrens_df = pd.DataFrame({
    'total_stories_count': ahrens_x,
    'ahrens_value': ahrens_y
})

# Dictionary to store correlation results for each test
correlation_results = {}

for test_id in speaker_story_mapping_df['test_id'].unique():
    # 2) Filter your data for this test_id and restrict to Ahrens' x-values
    test_df = speaker_story_mapping_df[
        speaker_story_mapping_df['test_id'] == test_id
    ].copy()
    test_df = test_df[test_df['total_stories_count'].isin(ahrens_x)]
    
    # 3) Merge on 'total_stories_count'
    merged_df = pd.merge(test_df, ahrens_df, on='total_stories_count', how='inner')
    
    # Make sure they're sorted by total_stories_count if you like
    merged_df.sort_values('total_stories_count', inplace=True)
    
    # 4) Compute Pearson correlation
    #    Compare your "correctly_assigned_stories_percentage"
    #    vs. "ahrens_value"
    if len(merged_df) > 1:
        r, p = pearsonr(
            merged_df['ahrens_value'],
            merged_df['correctly_assigned_stories_percentage']
        )
        correlation_results[test_id] = (r, p)
    else:
        # If there's only 1 overlapping point, correlation is undefined
        correlation_results[test_id] = (None, None)

# 5) Print the results
for test_id, (corr_val, p_val) in correlation_results.items():
    if corr_val is not None:
        print(f"{test_id}: Pearson r = {corr_val:.3f}, p-value = {p_val:.3g}")
    else:
        print(f"{test_id}: Not enough points for correlation.")


# In[184]:


ax = sns.boxplot(x='total_stories_count', y='correctly_assigned_stories_percentage', hue='test_id', data=speaker_story_mapping_df, palette="deep")
ax.set_xlabel("Total number of stories")
ax.set_ylabel("Correctly assigned stories [%]")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='Experimental condition', labels=['360° diotic', '360° binaural', 'CGI binaural', 'Ahrens et al. (2019)'])
d(ax)

save_fig(ax, "Correctly assigned stories per test", "plots/boxplot_speaker_story_mapping_percentage_per_test.pdf")
save_fig(ax, "Correctly assigned stories per test", "plots/boxplot_speaker_story_mapping_percentage_per_test.png")


# In[185]:


speaker_story_mapping_df_aov_mod = speaker_story_mapping_df[["subject_number", "test_id", "total_stories_count", "correctly_assigned_stories_percentage"]]
speaker_story_mapping_df_aov_mod['subject_number'] = (speaker_story_mapping_df_aov_mod.index // 9) + 1

d(speaker_story_mapping_df_aov_mod)

shapiro = pg.normality(speaker_story_mapping_df_aov_mod, dv="correctly_assigned_stories_percentage", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(speaker_story_mapping_df_aov_mod, dv="correctly_assigned_stories_percentage", group="test_id", method='levene', alpha=0.05)
d(levene)

# box_m = pg.box_m(speaker_story_mapping_df_aov_mod, dvs=["correctly_assigned_stories_percentage", "no_assigned_stories_percentage"], group="test_id", alpha=0.001)
# d(box_m)

aov = pg.mixed_anova(data=speaker_story_mapping_df_aov_mod, dv="correctly_assigned_stories_percentage", within="total_stories_count", subject="subject_number", between="test_id", correction='auto', effsize='np2')
d(aov)
aov.to_csv("data_eval/anova_speaker_story_mapping_percentage_per_test.csv")

post_hoc = pg.pairwise_tests(
    data=speaker_story_mapping_df_aov_mod,
    dv="correctly_assigned_stories_percentage",
    within="total_stories_count",
    between="test_id",
    subject="subject_number",
    padjust="bonf"
)
d(post_hoc)
post_hoc.to_csv("data_eval/post-hoc_story_mapping_percentage_per_test.csv")


# In[186]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(speaker_story_mapping_df_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(correctly_assigned_stories_percentage ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_story_mapping_percentage_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_story_mapping_percentage_per_test.csv")


# # Speaker-story-mapping performance (percentage of not assigned stories)

# In[187]:


ax = sns.lineplot(x='total_stories_count', y='no_assigned_stories_percentage', hue='test_id', data=speaker_story_mapping_df, palette="deep")
ax.set_ylabel("Not assigned stories [%]")
ax.set_xlabel("Total number of stories")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='Experimental condition', labels=['360° diotic', '360° binaural', 'CGI binaural'])
d(ax)
d(speaker_story_mapping_df.groupby(["test_id", "total_stories_count"])["no_assigned_stories_percentage"].mean().reset_index())

save_fig(ax, "", "plots/lineplot_speaker_story_mapping_percentage_notassignedstories_per_test.pdf")
save_fig(ax, "", "plots/lineplot_speaker_story_mapping_percentage_notassignedstories_per_test.png")


# In[188]:


# Compute deviation
deviation_list = []
num_chairs = 20

for idx, row in speaker_story_mapping_df.iterrows():
    deviation = 0
    for i in range(1, num_chairs + 1):
        story_col = f'chair_{i}.story'
        assigned_story_col = f'chair_{i}.assigned_story'
        actual_story = row[story_col]
        assigned_story = row[assigned_story_col]
        if actual_story != assigned_story and assigned_story != 11:
            for j in range(1, num_chairs + 1):
                if row[f'chair_{j}.story'] == assigned_story and i != j:
                    direct_distance = abs(i - j)
                    circular_distance = num_chairs - direct_distance
                    deviation += (min(direct_distance, circular_distance)) * 18  # to get deviation in degrees
                    break
    deviation_list.append(deviation)

speaker_story_mapping_df['deviation_count'] = deviation_list
d(speaker_story_mapping_df)
# speaker_story_mapping_df.to_excel("speaker_story_mapping_df.xlsx", sheet_name='speaker_story_mapping_df')


# # Total time needed

# In[189]:


unique_test_ids = speaker_story_mapping_df['test_id'].unique()
palette = sns.color_palette("deep", len(unique_test_ids))
sns.set_palette(palette)

ax = sns.lineplot(x='total_stories_count', y='total_time_s', hue='test_id', data=headrotation_df, palette=palette)

# Add the results from Ahrens et al. (2019)
ahrens_x = [2, 3, 4, 5, 6, 7, 8, 9, 10]
ahrens_y = [35, 40, 72, 85, 115, 120, 120, 120, 120]
ax.plot(ahrens_x, ahrens_y, color='gray', linestyle='-', linewidth=1)
ahrens_handle = Line2D([0], [0], color='gray', linestyle='-', linewidth=1)
handles, labels = ax.get_legend_handles_labels()
handles.append(ahrens_handle)
labels.append('Ahrens et al. (2019)')

ax.set_xlabel("Total number of stories")
ax.set_ylabel("Total time needed [s]")
ax.legend(handles=handles, title='Experimental condition', labels=['360° diotic', '360° binaural', 'CGI binaural', 'Ahrens et al. (2019)'])
d(ax)

save_fig(ax, "", "plots/lineplot_totaltimeneeded_per_test.pdf")
save_fig(ax, "", "plots/lineplot_totaltimeneeded_per_test.png")
d(headrotation_df.groupby(["test_id", "total_stories_count"])["total_time_s"].mean().reset_index())


# In[190]:


# Suppose your DataFrame is named 'df'
summary_df = (
    headrotation_df
    .groupby(['test_id', 'total_stories_count'])
    .agg(
        total_count=('total_time_s', 'size'),
        count_120=('total_time_s', lambda x: (x == 120).sum()),
        percentage_120=('total_time_s', lambda x: (x == 120).mean() * 100)
    )
)

# If you want a regular column index rather than a multiindex, you can do:
summary_df = summary_df.reset_index()

d(summary_df)


# In[191]:


# 1) Put Ahrens data in a DataFrame
ahrens_df = pd.DataFrame({
    'total_stories_count': ahrens_x,
    'ahrens_value': ahrens_y
})

# Dictionary to store correlation results for each test
correlation_results = {}

for test_id in headrotation_df['test_id'].unique():
    # 2) Filter your data for this test_id and restrict to Ahrens' x-values
    test_df = headrotation_df[
        headrotation_df['test_id'] == test_id
    ].copy()
    test_df = test_df[test_df['total_stories_count'].isin(ahrens_x)]
    
    # 3) Merge on 'total_stories_count'
    merged_df = pd.merge(test_df, ahrens_df, on='total_stories_count', how='inner')
    
    # Make sure they're sorted by total_stories_count if you like
    merged_df.sort_values('total_stories_count', inplace=True)
    
    # 4) Compute Pearson correlation
    #    Compare your "correctly_assigned_stories_percentage"
    #    vs. "ahrens_value"
    if len(merged_df) > 1:
        r, p = pearsonr(
            merged_df['ahrens_value'],
            merged_df['total_time_s']
        )
        correlation_results[test_id] = (r, p)
    else:
        # If there's only 1 overlapping point, correlation is undefined
        correlation_results[test_id] = (None, None)

# 5) Print the results
for test_id, (corr_val, p_val) in correlation_results.items():
    if corr_val is not None:
        print(f"{test_id}: Pearson r = {corr_val:.3f}, p-value = {p_val:.3g}")
    else:
        print(f"{test_id}: Not enough points for correlation.")


# In[192]:


headrotation_df_aov_mod = headrotation_df
headrotation_df_aov_mod['subject_number'] = (headrotation_df_aov_mod.index // 9) + 1

d(headrotation_df_aov_mod)

shapiro = pg.normality(headrotation_df_aov_mod, dv="total_time_s", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(headrotation_df_aov_mod, dv="total_time_s", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.mixed_anova(data=headrotation_df_aov_mod, dv="total_time_s", within="total_stories_count", subject="subject_number", between="test_id", correction='auto', effsize='np2')
d(aov)
aov.to_csv("data_eval/anova_total_time_needed_per_test.csv")

post_hoc = pg.pairwise_tests(
    data=headrotation_df_aov_mod,
    dv="total_time_s",
    within="total_stories_count",
    between="test_id",
    subject="subject_number",
    padjust="bonf"
)
d(post_hoc)
post_hoc.to_csv("data_eval/post-hoc_total_time_needed_per_test.csv")


# In[193]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(headrotation_df_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(total_time_s ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_total_time_needed_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_total_time_needed_per_test.csv")


# # Further data processing (ETL), distance between speakers

# In[194]:


# Compute pairwise distances between speakers [# of chairs]
pairwise_distances = []
num_chairs = 20
for idx, row in speaker_story_mapping_df.iterrows():
    distances = 0
    for i in range(1, num_chairs + 1):
        for j in range(i + 1, num_chairs + 1):
            story_i = row[f'chair_{i}.story']
            story_j = row[f'chair_{j}.story']
            if story_i != story_j and story_i != 11 and story_j != 11:
                subject_number = row["subject_number"]
                scene = row["scene"]
                direct_distance = abs(i - j)
                circular_distance = num_chairs - direct_distance
                distances += min(direct_distance, circular_distance) * 18
                # d(f"Subject: {subject_number}, Scene: {scene}, story_i: {i}, story_j: {j}, distances: {distances}")
                break
    pairwise_distances.append(distances)

speaker_story_mapping_df['distance_between_speakers'] = pairwise_distances
d(speaker_story_mapping_df)


# In[195]:


ax = sns.lineplot(x='total_stories_count', y='distance_between_speakers', hue='test_id', data=speaker_story_mapping_df, palette="deep")
ax.set_xlabel("Total number of stories")
ax.set_ylabel("Distance between active speakers [°]")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='Experimental condition', labels=['360° diotic', '360° binaural', 'CGI binaural'])
d(ax)
save_fig(ax, "", "plots/lineplot_distancebetweenspeakers_totalnumberofstories.pdf")
save_fig(ax, "", "plots/lineplot_distancebetweenspeakers_totalnumberofstories.png")


# # NASA TLX & listening effort

# In[196]:


columns_to_convert = ['subject_number', 'test_id', 'scene', 'rating']
nasatlx_df[columns_to_convert] = nasatlx_df[columns_to_convert].apply(pd.to_numeric)
d(nasatlx_df)


# In[197]:


columns_to_convert = ['subject_number', 'test_id', 'scene', 'rating']
nasatlx_df_reduced = nasatlx_df[(nasatlx_df['rating_type'] != 'nasatlx_mental_workload_score')]
nasatlx_df_reduced[columns_to_convert] = nasatlx_df_reduced[columns_to_convert].apply(pd.to_numeric)
d(nasatlx_df_reduced)


# In[198]:


speaker_story_mapping_df = speaker_story_mapping_df.apply(pd.to_numeric)

nasatlx_df_reduced_merged = nasatlx_df_reduced.merge(
    speaker_story_mapping_df[['test_id', 'scene', 'subject_number', 'total_stories_count']],
    on=['test_id', 'scene', 'subject_number'],
    how='left'
)
d(nasatlx_df_reduced_merged)

plt.figure(figsize=(5.8, 5))
ax = sns.lineplot(x='total_stories_count', y='rating', hue='rating_type', data=nasatlx_df_reduced_merged.loc[nasatlx_df_reduced_merged['test_id'] == 1], palette="husl")
ax.set_xlabel("Total number of stories")
ax.set_ylabel("Score")
ax.set_ylim(0, 100)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='360° diotic - Mental Load Scores', labels=['NASA RTLX effort', 'NASA RTLX frustration', 'NASA RTLX mental demand', 'NASA RTLX performance', 'NASA RTLX temporal demand', 'Listening effort'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
d(ax)
save_fig(ax, "", "plots/lineplot_nasatlx_test_1.pdf")
save_fig(ax, "", "plots/lineplot_nasatlx_test_1.png")


# In[199]:


plt.figure(figsize=(5.8, 5))
ax = sns.lineplot(x='total_stories_count', y='rating', hue='rating_type', data=nasatlx_df_reduced_merged.loc[nasatlx_df_reduced_merged['test_id'] == 2], palette="husl")
ax.set_xlabel("Total number of stories")
ax.set_ylabel("Score")
ax.set_ylim(0, 100)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='360° binaural - Mental Load Scores', labels=['NASA RTLX effort', 'NASA RTLX frustration', 'NASA RTLX mental demand', 'NASA RTLX performance', 'NASA RTLX temporal demand', 'Listening effort'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
d(ax)
save_fig(ax, "", "plots/lineplot_nasatlx_test_2.pdf")
save_fig(ax, "", "plots/lineplot_nasatlx_test_2.png")


# In[200]:


plt.figure(figsize=(5.8, 5))
ax = sns.lineplot(x='total_stories_count', y='rating', hue='rating_type', data=nasatlx_df_reduced_merged.loc[nasatlx_df_reduced_merged['test_id'] == 3], palette="husl")
ax.set_xlabel("Total number of stories")
ax.set_ylabel("Score")
ax.set_ylim(0, 100)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='CGI binaural - Mental Load Scores', labels=['NASA RTLX effort', 'NASA RTLX frustration', 'NASA RTLX mental demand', 'NASA RTLX performance', 'NASA RTLX temporal demand', 'Listening effort'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
d(ax)
save_fig(ax, "", "plots/lineplot_nasatlx_test_3.pdf")
save_fig(ax, "", "plots/lineplot_nasatlx_test_3.png")


# In[201]:


pairwise_results = pg.pairwise_tests(
    data=nasatlx_df,
    dv='rating',
    within='rating_type',
    subject='subject_number',
    padjust='bonf'
)
d(pairwise_results)
pairwise_results.to_csv("data_eval/pairwisetests_nasatlx_listeningeffort.csv")

nasatlx_df_merged = nasatlx_df.merge(
    speaker_story_mapping_df[['test_id', 'scene', 'subject_number', 'total_stories_count']],
    on=['test_id', 'scene', 'subject_number'],
    how='left'
)

nasatlx_df_merged_aov_mod = nasatlx_df_merged.pivot_table(
    index=['subject_number', 'test_id', 'total_stories_count'],
    columns='rating_type',
    values='rating'
).reset_index()
nasatlx_df_merged_aov_mod['subject_number'] = (nasatlx_df_merged_aov_mod.index // 9) + 1
d(nasatlx_df_merged_aov_mod)

shapiro = pg.normality(nasatlx_df_merged_aov_mod, dv="nasatlx_mental_workload_score", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(nasatlx_df_merged_aov_mod, dv="nasatlx_mental_workload_score", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.mixed_anova(data=nasatlx_df_merged_aov_mod, dv="nasatlx_mental_workload_score", within="total_stories_count", subject="subject_number", between="test_id", correction='auto', effsize='np2')
d(aov)
aov.to_csv("data_eval/anova_nasatlx_mental_workload_per_test.csv")

post_hoc = pg.pairwise_tests(
    data=nasatlx_df_merged_aov_mod,
    dv="nasatlx_mental_workload_score",
    within="total_stories_count",
    between="test_id",
    subject="subject_number",
    padjust="bonf"
)
d(post_hoc)
post_hoc.to_csv("data_eval/post-hoc_nasatlx_mental_workload_per_test.csv")


# In[202]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(nasatlx_df_merged_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(nasatlx_mental_workload_score ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_nasatlx_mental_workload_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_nasatlx_mental_workload_per_test.csv")


# In[203]:


shapiro = pg.normality(nasatlx_df_merged_aov_mod, dv="nasatlx_effort", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(nasatlx_df_merged_aov_mod, dv="nasatlx_effort", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.mixed_anova(data=nasatlx_df_merged_aov_mod, dv="nasatlx_effort", within="total_stories_count", subject="subject_number", between="test_id", correction='auto', effsize='np2')
d(aov)
aov.to_csv("data_eval/anova_nasatlx_effort_per_test.csv")

post_hoc = pg.pairwise_tests(
    data=nasatlx_df_merged_aov_mod,
    dv="nasatlx_effort",
    within="total_stories_count",
    between="test_id",
    subject="subject_number",
    padjust="bonf"
)
d(post_hoc)
post_hoc.to_csv("data_eval/post-hoc_nasatlx_effort_per_test.csv")


# In[204]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(nasatlx_df_merged_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(nasatlx_effort ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_nasatlx_effort_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_nasatlx_effort_per_test.csv")


# In[205]:


shapiro = pg.normality(nasatlx_df_merged_aov_mod, dv="nasatlx_frustration", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(nasatlx_df_merged_aov_mod, dv="nasatlx_frustration", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.mixed_anova(data=nasatlx_df_merged_aov_mod, dv="nasatlx_frustration", within="total_stories_count", subject="subject_number", between="test_id", correction='auto', effsize='np2')
d(aov)
aov.to_csv("data_eval/anova_nasatlx_frustration_per_test.csv")

post_hoc = pg.pairwise_tests(
    data=nasatlx_df_merged_aov_mod,
    dv="nasatlx_frustration",
    within="total_stories_count",
    between="test_id",
    subject="subject_number",
    padjust="bonf"
)
d(post_hoc)
post_hoc.to_csv("data_eval/post-hoc_nasatlx_frustration_per_test.csv")


# In[206]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(nasatlx_df_merged_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(nasatlx_frustration ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_nasatlx_frustration_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_nasatlx_frustration_per_test.csv")


# In[207]:


shapiro = pg.normality(nasatlx_df_merged_aov_mod, dv="nasatlx_mental_demand", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(nasatlx_df_merged_aov_mod, dv="nasatlx_mental_demand", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.mixed_anova(data=nasatlx_df_merged_aov_mod, dv="nasatlx_mental_demand", within="total_stories_count", subject="subject_number", between="test_id", correction='auto', effsize='np2')
d(aov)
aov.to_csv("data_eval/anova_nasatlx_mental_demand_per_test.csv")

post_hoc = pg.pairwise_tests(
    data=nasatlx_df_merged_aov_mod,
    dv="nasatlx_mental_demand",
    within="total_stories_count",
    between="test_id",
    subject="subject_number",
    padjust="bonf"
)
d(post_hoc)
post_hoc.to_csv("data_eval/post-hoc_nasatlx_mental_demand_per_test.csv")


# In[208]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(nasatlx_df_merged_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(nasatlx_mental_demand ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_nasatlx_mental_demand_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_nasatlx_mental_demand_per_test.csv")


# In[209]:


shapiro = pg.normality(nasatlx_df_merged_aov_mod, dv="nasatlx_performance", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(nasatlx_df_merged_aov_mod, dv="nasatlx_performance", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.mixed_anova(data=nasatlx_df_merged_aov_mod, dv="nasatlx_performance", within="total_stories_count", subject="subject_number", between="test_id", correction='auto', effsize='np2')
d(aov)
aov.to_csv("data_eval/anova_nasatlx_performance_per_test.csv")

post_hoc = pg.pairwise_tests(
    data=nasatlx_df_merged_aov_mod,
    dv="nasatlx_performance",
    within="total_stories_count",
    between="test_id",
    subject="subject_number",
    padjust="bonf"
)
d(post_hoc)
post_hoc.to_csv("data_eval/post-hoc_nasatlx_performance_per_test.csv")


# In[210]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(nasatlx_df_merged_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(nasatlx_performance ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_nasatlx_performance_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_nasatlx_performance_per_test.csv")


# In[211]:


shapiro = pg.normality(nasatlx_df_merged_aov_mod, dv="nasatlx_temporal_demand", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(nasatlx_df_merged_aov_mod, dv="nasatlx_temporal_demand", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.mixed_anova(data=nasatlx_df_merged_aov_mod, dv="nasatlx_temporal_demand", within="total_stories_count", subject="subject_number", between="test_id", correction='auto', effsize='np2')
d(aov)
aov.to_csv("data_eval/anova_nasatlx_temporal_demand_per_test.csv")

post_hoc = pg.pairwise_tests(
    data=nasatlx_df_merged_aov_mod,
    dv="nasatlx_temporal_demand",
    within="total_stories_count",
    between="test_id",
    subject="subject_number",
    padjust="bonf"
)
d(post_hoc)
post_hoc.to_csv("data_eval/post-hoc_nasatlx_temporal_demand_per_test.csv")


# In[212]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(nasatlx_df_merged_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(nasatlx_temporal_demand ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_nasatlx_temporal_demand_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_nasatlx_temporal_demand_per_test.csv")


# In[213]:


shapiro = pg.normality(nasatlx_df_merged_aov_mod, dv="listening_effort", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(nasatlx_df_merged_aov_mod, dv="listening_effort", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.mixed_anova(data=nasatlx_df_merged_aov_mod, dv="listening_effort", within="total_stories_count", subject="subject_number", between="test_id", correction='auto', effsize='np2')
d(aov)
aov.to_csv("data_eval/anova_listening_effort_per_test.csv")

post_hoc = pg.pairwise_tests(
    data=nasatlx_df_merged_aov_mod,
    dv="listening_effort",
    within="total_stories_count",
    between="test_id",
    subject="subject_number",
    padjust="bonf"
)
d(post_hoc)
post_hoc.to_csv("data_eval/post-hoc_listening_effort_per_test.csv")


# In[214]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(nasatlx_df_merged_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(listening_effort ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_listening_effort_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_listening_effort_per_test.csv")


# # Total degrees explored

# In[215]:


d(headrotation_df)
headrotation_df_melted = headrotation_df.melt(id_vars=["subject_number", "test_id", "total_stories_count", "total_time_s"], value_vars=["total_pitch_explored", "total_yaw_explored", "total_degrees_explored"],
                    var_name="exploration_type", value_name="degrees_explored")
d(headrotation_df_melted)


# In[216]:


ax = sns.lineplot(x='total_stories_count', y='total_yaw_explored', hue='test_id', data=headrotation_df, palette="deep")
ax.set_xlabel("Total number of stories")
ax.set_ylabel("Total yaw degrees explored [°]")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='Experimental condition', labels=['360° diotic', '360° binaural', 'CGI binaural'])
d(ax)
save_fig(ax, "", "plots/lineplot_total_yaw_explored.pdf")
save_fig(ax, "", "plots/lineplot_total_yaw_explored.png")
d(headrotation_df.groupby(["test_id", "total_stories_count"])["total_yaw_explored"].mean().reset_index())


# In[217]:


d(headrotation_df_aov_mod)

shapiro = pg.normality(headrotation_df_aov_mod, dv="total_yaw_explored", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(headrotation_df_aov_mod, dv="total_yaw_explored", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.mixed_anova(data=headrotation_df_aov_mod, dv="total_yaw_explored", within="total_stories_count", subject="subject_number", between="test_id", correction='auto', effsize='np2')
d(aov)
aov.to_csv("data_eval/anova_total_yaw_explored_per_test.csv")

post_hoc = pg.pairwise_tests(
    data=headrotation_df_aov_mod,
    dv="total_yaw_explored",
    within="total_stories_count",
    between="test_id",
    subject="subject_number",
    padjust="bonf"
)
d(post_hoc)
post_hoc.to_csv("data_eval/post-hoc_total_yaw_explored_per_test.csv")


# In[218]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(headrotation_df_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(total_yaw_explored ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_total_yaw_explored_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_total_yaw_explored_per_test.csv")


# # Total number of yaw direction changes

# In[219]:


ax = sns.lineplot(x='total_stories_count', y='number_yaw_direction_changes', hue='test_id', data=headrotation_df, palette="deep")
ax.set_xlabel("Total number of stories")
ax.set_ylabel("Total number of yaw direction changes")

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='Experimental condition', labels=['360° diotic', '360° binaural', 'CGI binaural'])
d(ax)
save_fig(ax, "", "plots/lineplot_total_yaw_direction_changes.pdf")
save_fig(ax, "", "plots/lineplot_total_yaw_direction_changes.png")
d(headrotation_df.groupby(["test_id", "total_stories_count"])["number_yaw_direction_changes"].mean().reset_index())


# # Number of yaw direction changes per second

# In[220]:


ax = sns.lineplot(x='total_stories_count', y='number_yaw_direction_changes_per_s', hue='test_id', data=headrotation_df, palette="deep")
ax.set_xlabel("Total number of stories")
ax.set_ylabel("Number of yaw direction changes per second")

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='Experimental condition', labels=['360° diotic', '360° binaural', 'CGI binaural'])
d(ax)
save_fig(ax, "", "plots/lineplot_number_yaw_direction_changes_per_s.pdf")
save_fig(ax, "", "plots/lineplot_number_yaw_direction_changes_per_s.png")
d(headrotation_df.groupby(["test_id", "total_stories_count"])["number_yaw_direction_changes_per_s"].mean().reset_index())


# In[221]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(headrotation_df)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(number_yaw_direction_changes_per_s ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_number_yaw_direction_changes_per_s_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_number_yaw_direction_changes_per_s_per_test.csv")


# In[222]:


d(headrotation_df_aov_mod)

shapiro = pg.normality(headrotation_df_aov_mod, dv="number_yaw_direction_changes", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(headrotation_df_aov_mod, dv="number_yaw_direction_changes", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.mixed_anova(data=headrotation_df_aov_mod, dv="number_yaw_direction_changes", within="total_stories_count", subject="subject_number", between="test_id", correction='auto', effsize='np2')
d(aov)
aov.to_csv("data_eval/anova_total_yaw_direction_changes_per_test.csv")

post_hoc = pg.pairwise_tests(
    data=headrotation_df_aov_mod,
    dv="number_yaw_direction_changes",
    within="total_stories_count",
    between="test_id",
    subject="subject_number",
    padjust="bonf"
)
d(post_hoc)
post_hoc.to_csv("data_eval/post-hoc_total_yaw_direction_changes_per_test.csv")


# In[223]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(headrotation_df_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(number_yaw_direction_changes ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_total_yaw_direction_changes_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_total_yaw_direction_changes_per_test.csv")


# # Further data processing (ETL)

# In[224]:


# Unity - behaviour CSV (timestamp, controller interaction, head rotation, etc.)
# Including all pitch_yaw_roll data

def map_yaw_to_chair(yaw_value):
    chair_index = int(yaw_value // 18)
    chair_index += 10  # chair_index corresponds to yaw = 0 (chair 10 here)
    return chair_index % 20 if chair_index % 20 != 0 else 20

headrotation_big_df = pd.DataFrame()

for test_id, test_name in zip(test_ids, test_names):
    for file in glob.glob('../subjective_data/{}/Unity/*.csv'.format(test_name)):
        if not "training" in file:
            subject = Path(file).stem.split('_')[0]
            total_stories = Path(file).stem.split('_')[1]
            df = pd.read_csv(file, index_col=False)
            total_time_s = (pd.to_datetime(df['TimeStamp'].iloc[-1], format='%d.%m.%Y %H:%M:%S') - pd.to_datetime(df['TimeStamp'].iloc[0], format='%d.%m.%Y %H:%M:%S')).total_seconds()
            chairs_time_s = [0] * 20
            for idx, row in df.iterrows():
                yaw_value = row["VRCameraYaw"]
                chair_index = map_yaw_to_chair(yaw_value) - 1
                chairs_time_s[chair_index] += 1
            chairs_time_s = [time_s / len(df) * total_time_s for time_s in chairs_time_s]
            tmp_dict = {
                'subject_number': [subject],
                'test_id': [test_id],
                'total_stories_count': [total_stories],
                'total_time_s': [total_time_s],
            }
            tmp_dict.update({f'chair_{i+1}.time_s': [time_s] for i, time_s in enumerate(chairs_time_s)})
            tmp_df = pd.DataFrame(tmp_dict)
            headrotation_big_df = pd.concat([headrotation_big_df, tmp_df], ignore_index=True)
headrotation_big_df = headrotation_big_df.apply(pd.to_numeric)
d(headrotation_big_df)
# headrotation_big_df.to_excel("headrotation_big_df.xlsx", sheet_name='headrotation_big_df')


# In[225]:


columns_to_merge = ['test_id', 'subject_number', 'total_stories_count']
speaker_story_mapping_df = speaker_story_mapping_df.apply(pd.to_numeric)
headrotation_big_speaker_story_mapping_df = pd.merge(headrotation_big_df, speaker_story_mapping_df, on=columns_to_merge)
columns_to_drop = [
    'chair_1.speaker', 'chair_1.assigned_story',
    'chair_2.speaker', 'chair_2.assigned_story',
    'chair_3.speaker', 'chair_3.assigned_story',
    'chair_4.speaker', 'chair_4.assigned_story',
    'chair_5.speaker', 'chair_5.assigned_story',
    'chair_6.speaker', 'chair_6.assigned_story',
    'chair_7.speaker', 'chair_7.assigned_story',
    'chair_8.speaker', 'chair_8.assigned_story',
    'chair_9.speaker', 'chair_9.assigned_story',
    'chair_10.speaker', 'chair_10.assigned_story',
    'chair_11.speaker', 'chair_11.assigned_story',
    'chair_12.speaker', 'chair_12.assigned_story',
    'chair_13.speaker', 'chair_13.assigned_story',
    'chair_14.speaker', 'chair_14.assigned_story',
    'chair_15.speaker', 'chair_15.assigned_story',
    'chair_16.speaker', 'chair_16.assigned_story',
    'chair_17.speaker', 'chair_17.assigned_story',
    'chair_18.speaker', 'chair_18.assigned_story',
    'chair_19.speaker', 'chair_19.assigned_story',
    'chair_20.speaker', 'chair_20.assigned_story'
]
headrotation_big_speaker_story_mapping_df = headrotation_big_speaker_story_mapping_df.drop(columns=columns_to_drop)
d(headrotation_big_speaker_story_mapping_df)


# # Proportion of time spent watching active speakers

# In[226]:


headrotation_big_speaker_story_mapping_df['time_spent_watching_active_speakers'] = 0

# Loop through each row to calculate the time spent watching active speakers
for idx, row in headrotation_big_speaker_story_mapping_df.iterrows():
    time_spent = 0
    for chair_num in range(1, 21):
        story_col = f'chair_{chair_num}.story'
        time_col = f'chair_{chair_num}.time_s'
        if row[story_col] != 11:  # Check if the story is not 11
            time_spent += row[time_col]  # Add the time spent on this chair
            
    headrotation_big_speaker_story_mapping_df.at[idx, 'time_spent_watching_active_speakers'] = time_spent
    
headrotation_big_speaker_story_mapping_df['time_spent_watching_active_speakers_percentage'] = (
    headrotation_big_speaker_story_mapping_df['time_spent_watching_active_speakers'] / headrotation_big_speaker_story_mapping_df['total_time_s'] * 100
)
d(headrotation_big_speaker_story_mapping_df)


# In[227]:


ax = sns.lineplot(x='total_stories_count', y='time_spent_watching_active_speakers_percentage', hue='test_id', data=headrotation_big_speaker_story_mapping_df, palette="deep")
ax.set_xlabel("Total number of stories")
ax.set_ylabel("Proportion of time spent watching active speakers [%]")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='Experimental condition', labels=['360° diotic', '360° binaural', 'CGI binaural'])
d(ax)

save_fig(ax, "", "plots/lineplot_timespentwatchingactivespeakerspercentage_per_test.pdf")
save_fig(ax, "", "plots/lineplot_timespentwatchingactivespeakerspercentage_per_test.png")
d(headrotation_big_speaker_story_mapping_df.groupby(["test_id", "total_stories_count"])["time_spent_watching_active_speakers_percentage"].mean().reset_index())


# # Time spent watching active speakers

# In[228]:


ax = sns.lineplot(x='total_stories_count', y='time_spent_watching_active_speakers', hue='test_id', data=headrotation_big_speaker_story_mapping_df, palette="deep")
ax.set_xlabel("Total number of stories")
ax.set_ylabel("Time spent watching active speakers [s]")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title='Experimental condition', labels=['360° diotic', '360° binaural', 'CGI binaural'])
d(ax)

save_fig(ax, "", "plots/lineplot_timespentwatchingactivespeakers_per_test.pdf")
save_fig(ax, "", "plots/lineplot_timespentwatchingactivespeakers_per_test.png")
d(headrotation_big_speaker_story_mapping_df.groupby(["test_id", "total_stories_count"])["time_spent_watching_active_speakers"].mean().reset_index())


# In[229]:


headrotation_big_speaker_story_mapping_df_aov_mod = headrotation_big_speaker_story_mapping_df[["subject_number", "test_id", "total_stories_count", "time_spent_watching_active_speakers_percentage"]]
headrotation_big_speaker_story_mapping_df_aov_mod['subject_number'] = (headrotation_big_speaker_story_mapping_df_aov_mod.index // 9) + 1

d(headrotation_big_speaker_story_mapping_df_aov_mod)

shapiro = pg.normality(headrotation_big_speaker_story_mapping_df_aov_mod, dv="time_spent_watching_active_speakers_percentage", group="test_id", method='shapiro', alpha=0.05)
d(shapiro)

levene = pg.homoscedasticity(headrotation_big_speaker_story_mapping_df_aov_mod, dv="time_spent_watching_active_speakers_percentage", group="test_id", method='levene', alpha=0.05)
d(levene)

aov = pg.mixed_anova(data=headrotation_big_speaker_story_mapping_df_aov_mod, dv="time_spent_watching_active_speakers_percentage", within="total_stories_count", subject="subject_number", between="test_id", correction='auto', effsize='np2')
d(aov)
aov.to_csv("data_eval/anova_time_spent_watching_active_speakers_per_test.csv")

post_hoc = pg.pairwise_tests(
    data=headrotation_big_speaker_story_mapping_df_aov_mod,
    dv="time_spent_watching_active_speakers_percentage",
    within="total_stories_count",
    between="test_id",
    subject="subject_number",
    padjust="bonf"
)
d(post_hoc)
post_hoc.to_csv("data_eval/post-hoc_time_spent_watching_active_speakers_per_test.csv")


# In[230]:


pandas2ri.activate()
r_df = pandas2ri.py2rpy(headrotation_big_speaker_story_mapping_df_aov_mod)
ro.globalenv['df'] = r_df
artool = importr('ARTool')
# Change the following to mixed somehow
ro.r('''
df$test_id <- factor(df$test_id)
df$total_stories_count <- factor(df$total_stories_count)
library(ARTool)
m <- art(time_spent_watching_active_speakers_percentage ~ test_id * total_stories_count  + (1|subject_number), data=df)
''')

anova_res = ro.r('anova(m)')
d(pandas2ri.rpy2py(anova_res))
pandas2ri.rpy2py(anova_res).to_csv("data_eval/art_anova_time_spent_watching_active_speakers_per_test.csv")
post_hoc_res = ro.r('''
res <- art.con(m, "test_id", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
d(pandas2ri.rpy2py(post_hoc_res))

post_hoc_res = ro.r('''
res <- art.con(m, "test_id:total_stories_count", adjust="none")
df_res <- as.data.frame(res)
df_res
''')
post_hoc_res_df = pandas2ri.rpy2py(post_hoc_res)
desired_contrasts = ['1,2 - 2,2',
'1,2 - 3,2',
'2,2 - 3,2',
 '1,3 - 2,3',
 '1,3 - 3,3',
 '2,3 - 3,3',
 '1,4 - 2,4',
 '1,4 - 3,4',
'2,4 - 3,4',
 '1,5 - 2,5',
 '1,5 - 3,5',
'2,5 - 3,5',
 '1,6 - 2,6',
 '1,6 - 3,6',
'2,6 - 3,6',
 '1,7 - 2,7',
 '1,7 - 3,7',
'2,7 - 3,7',
 '1,8 - 2,8',
 '1,8 - 3,8',
'2,8 - 3,8',
 '1,9 - 2,9',
 '1,9 - 3,9',
'2,9 - 3,9',
 '1,10 - 2,10',
 '1,10 - 3,10',
'2,10 - 3,10'
]
post_hoc_res_df_filtered = post_hoc_res_df[post_hoc_res_df['contrast'].isin(desired_contrasts)].copy()
pvals = post_hoc_res_df_filtered['p.value'].values
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
post_hoc_res_df_filtered['p.value_corr'] = pvals_corrected

# sort "contrast" 
post_hoc_res_df_filtered['contrast'] = pd.Categorical(
    post_hoc_res_df_filtered['contrast'],
    categories=desired_contrasts,
    ordered=True
)
post_hoc_res_df_filtered.sort_values('contrast', inplace=True)

d(post_hoc_res_df_filtered)
post_hoc_res_df_filtered.to_csv("data_eval/art_post-hoc_time_spent_watching_active_speakers_per_test.csv")


# # Correlation matrices

# In[231]:


columns_to_merge = ['test_id', 'subject_number', 'total_stories_count']
for col in columns_to_merge:
    headrotation_df[col] = headrotation_df[col].astype(int)
    speaker_story_mapping_df[col] = speaker_story_mapping_df[col].astype(int)
headrotation_speaker_story_mapping_df = pd.merge(headrotation_df, speaker_story_mapping_df, on=columns_to_merge)
headrotation_speaker_story_mapping_nasatlx_df = pd.merge(headrotation_speaker_story_mapping_df, nasatlx_df, on=['test_id', 'scene', 'subject_number'])
# d(headrotation_speaker_story_mapping_nasatlx_df)

columns_to_drop = [
    'chair_1.speaker', 'chair_1.story', 'chair_1.assigned_story',
    'chair_2.speaker', 'chair_2.story', 'chair_2.assigned_story',
    'chair_3.speaker', 'chair_3.story', 'chair_3.assigned_story',
    'chair_4.speaker', 'chair_4.story', 'chair_4.assigned_story',
    'chair_5.speaker', 'chair_5.story', 'chair_5.assigned_story',
    'chair_6.speaker', 'chair_6.story', 'chair_6.assigned_story',
    'chair_7.speaker', 'chair_7.story', 'chair_7.assigned_story',
    'chair_8.speaker', 'chair_8.story', 'chair_8.assigned_story',
    'chair_9.speaker', 'chair_9.story', 'chair_9.assigned_story',
    'chair_10.speaker', 'chair_10.story', 'chair_10.assigned_story',
    'chair_11.speaker', 'chair_11.story', 'chair_11.assigned_story',
    'chair_12.speaker', 'chair_12.story', 'chair_12.assigned_story',
    'chair_13.speaker', 'chair_13.story', 'chair_13.assigned_story',
    'chair_14.speaker', 'chair_14.story', 'chair_14.assigned_story',
    'chair_15.speaker', 'chair_15.story', 'chair_15.assigned_story',
    'chair_16.speaker', 'chair_16.story', 'chair_16.assigned_story',
    'chair_17.speaker', 'chair_17.story', 'chair_17.assigned_story',
    'chair_18.speaker', 'chair_18.story', 'chair_18.assigned_story',
    'chair_19.speaker', 'chair_19.story', 'chair_19.assigned_story',
    'chair_20.speaker', 'chair_20.story', 'chair_20.assigned_story'
]
headrotation_speaker_story_mapping_nasatlx_df_small = headrotation_speaker_story_mapping_nasatlx_df.drop(columns=columns_to_drop)
# d(headrotation_speaker_story_mapping_nasatlx_df_small)

weinstein_df_reduced = weinstein_df.loc[:, ['subject_number', 'test_id', 'mean_weinstein_score']]
# d(weinstein_df_reduced)

headrotation_speaker_story_mapping_nasatlx_weinstein_df_small = pd.merge(headrotation_speaker_story_mapping_nasatlx_df_small, weinstein_df_reduced, on=['test_id', 'subject_number'])
headrotation_speaker_story_mapping_nasatlx_weinstein_df_small = pd.merge(headrotation_speaker_story_mapping_nasatlx_weinstein_df_small, headrotation_big_speaker_story_mapping_df[['test_id', 'subject_number', 'total_stories_count', 'time_spent_watching_active_speakers']], on=['test_id', 'subject_number', 'total_stories_count'])
headrotation_speaker_story_mapping_nasatlx_weinstein_ssq_df_small = pd.merge(headrotation_speaker_story_mapping_nasatlx_weinstein_df_small, post_ssq_df[['test_id', 'subject_number', 'TS']].apply(pd.to_numeric), on=['test_id', 'subject_number'])
d(headrotation_speaker_story_mapping_nasatlx_weinstein_ssq_df_small)


# In[232]:


headrotation_speaker_story_mapping_nasatlx_weinstein_ssq_df_small_pivoted = headrotation_speaker_story_mapping_nasatlx_weinstein_ssq_df_small.pivot_table(index=['total_time_s', 'time_spent_watching_active_speakers', 'total_stories_count', 'total_pitch_explored', 'total_yaw_explored', 'total_degrees_explored', 'number_yaw_direction_changes', 'number_yaw_direction_changes_per_s', 'correctly_assigned_stories_percentage', 'no_assigned_stories_percentage', 'mean_weinstein_score', 'TS', 'deviation_count', 'distance_between_speakers'],
                          columns='rating_type', values='rating').reset_index()
d(headrotation_speaker_story_mapping_nasatlx_weinstein_ssq_df_small_pivoted)
correlation_matrix = headrotation_speaker_story_mapping_nasatlx_weinstein_ssq_df_small_pivoted.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
plt.figure(figsize=(9, 9))
ax = sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 8})
ax.set_xlabel("Dimension")
ax.set_yticklabels(["Total time needed", "Proportion of time spent watching active speakers", "Total number of stories", "Total pitch explored", "Total yaw explored", "Total degrees explored", "Total number of yaw direction changes", "Number of yaw direction changes per second", "Correctly assigned stories [%]", "Not assigned stories [%]", "Mean Weinstein score", "SSQ (after) TS score", "Deviation [°]", "Distance between active speakers [°]", "Listening effort", "NASA RTLX effort", "NASA RTLX frustration", "NASA RTLX mental demand", "NASA RTLX mental workload score", "NASA RTLX performance", "NASA RTLX temporal demand"])
ax.set_ylabel("Dimension")
ax.set_xticklabels(["Total time needed", "Proportion of time spent watching active speakers", "Total number of stories", "Total pitch explored", "Total yaw explored", "Total degrees explored", "Total number of yaw direction changes", "Number of yaw direction changes per second", "Correctly assigned stories [%]", "Not assigned stories [%]", "Mean Weinstein score", "SSQ (after) TS score", "Deviation [°]", "Distance between active speakers [°]", "Listening effort", "NASA RTLX effort", "NASA RTLX frustration", "NASA RTLX mental demand", "NASA RTLX mental workload score", "NASA RTLX performance", "NASA RTLX temporal demand"])
d(ax)

save_fig(ax, "", "plots/correlation_matrix.pdf")
save_fig(ax, "", "plots/correlation_matrix.png")

