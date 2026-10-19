# Streamlit frontend: pick track + case, view the Reliability Report.
from __future__ import annotations
import os
import requests
import streamlit as st
BACKEND = os.environ.get('NEUROSHIELD_BACKEND', 'http://localhost:8000')
st.set_page_config(page_title='NeuroShield', layout='wide')
st.title('NeuroShield - Medical AI Reliability Report')
st.caption('Research prototype. Not a clinically validated diagnostic system.')
try:
    tracks = requests.get(f'{BACKEND}/health', timeout=5).json().get('tracks', [])
except Exception:
    tracks = []
    st.warning(f'Backend not reachable at {BACKEND}. Start it with uvicorn first.')
track = st.selectbox('Track', tracks)
case_id = st.text_input('Case ID (subject or case identifier)')
if st.button('Run reliability evaluation') and track and case_id:
    with st.spinner('Running...'):
        r = requests.post(f'{BACKEND}/run', json={'track': track, 'case_id': case_id}, timeout=600)
    if r.status_code != 200:
        st.error(r.json().get('detail', 'Run failed'))
    else:
        rep = r.json()
        verdict = rep['decision']
        (st.success if verdict.startswith('ACCEPT') else st.error)(f"{verdict} - reliability: {rep['reliability']}")
        cols = st.columns(3)
        cols[0].metric('Confidence', f"{rep['confidence']:.3f}")
        cols[1].metric('Uncertainty', rep['uncertainty'].get('level', '?'))
        cols[2].metric('OOD', rep['ood'].get('level', '?'))
        st.subheader('Signals')
        st.json({k: rep[k] for k in ('uncertainty', 'calibration', 'ood', 'counterfactual_stability', 'model_agreement')})
        if rep.get('decision_reasons'):
            st.subheader('Why this decision')
            for reason in rep['decision_reasons']:
                st.write('-', reason)
        if rep.get('explanation'):
            st.subheader('Evidence-grounded explanation')
            st.write(rep['explanation'])
            with st.expander('Retrieved evidence sources'):
                st.json(rep.get('explanation_evidence', []))
        st.caption(rep['disclaimer'])
