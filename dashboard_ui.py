"""Presentation components for the Air Quality Lab Streamlit application."""
from html import escape
from pathlib import Path

import altair as alt
import streamlit as st

COLORS = ["#0D8177", "#5074B8", "#CC8744", "#8064AA", "#718797"]
INK, GRID_COLOR, PAPER = "#163044", "#DCE5EB", "#FFFFFF"


def init_style():
    css = (Path(__file__).resolve().parent / "assets" / "dashboard.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sidebar_brand():
    st.sidebar.markdown(
        '''<div class="aq-brand"><div class="aq-brand-icon">
        <svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
        <path d="M4 11h15a4 4 0 1 0-4-4M4 16h22a3 3 0 1 1-3 3M4 21h10a4 4 0 1 1-4 4"
              stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></div>
        <div><div class="aq-brand-title">Air Quality Lab</div>
        <div class="aq-brand-sub">KELOMPOK 09 · TEKNIK ELEKTRO</div></div></div>
        <div class="aq-side-section">Menu eksplorasi</div>''', unsafe_allow_html=True)


def sidebar_status(source, rows):
    st.sidebar.markdown(
        f'''<div class="aq-side-status"><b><span class="aq-dot"></span>Model siap digunakan</b>
        <p>{escape(source)} · {rows:,} jam<br>Horizon +1 jam · mg/m³<br>Data historis, bukan sensor live.</p></div>''',
        unsafe_allow_html=True)


def hero():
    st.markdown('''
    <section class="aq-hero" aria-label="Ringkasan proyek">
      <div class="aq-hero-top">
        <span class="aq-eyebrow">AI Lab / TE2514031 / Gasal 2026–2027</span>
        <span class="aq-badge">DATA HISTORIS · MODEL TERLATIH</span>
      </div>
      <div class="aq-hero-grid"><div>
        <h1>Forecasting<br>Kualitas Udara</h1>
        <p>Prakiraan karbon monoksida berbasis kecerdasan buatan hibrida.
        Pelajari data, bandingkan empat model, dan jalankan simulasi dari riwayat sensor.</p>
        <div class="aq-pills"><span class="aq-pill">GA · Seleksi fitur</span>
        <span class="aq-pill">PSO · Optimasi</span><span class="aq-pill">RNN · Forecast</span>
        <span class="aq-pill">Fuzzy · Keputusan</span></div>
      </div><div class="aq-hero-number"><b>CO</b><span>Karbon monoksida<br>Target t + 1 jam<br>Satuan mg/m³</span></div></div>
    </section>''', unsafe_allow_html=True)


def page_header(index, subtitle, title, description):
    st.markdown(f'<div class="aq-page-breadcrumb"><div class="aq-eyebrow">MODUL {index + 1:02d} / {escape(subtitle)}</div>'
                '<span class="aq-badge">AIR QUALITY LAB</span></div>', unsafe_allow_html=True)
    st.title(title)
    if description:
        st.caption(description)


def note(title, body, tone="info"):
    css_class = {"info": "", "good": "good", "warn": "warn"}.get(tone, "")
    st.markdown(f'<div class="aq-note {css_class}"><strong>{escape(title)}</strong>'
                f'<span>{escape(body)}</span></div>', unsafe_allow_html=True)


def method_grid(items):
    markup = ['<div class="aq-method-grid">']
    for kicker, title, body, value in items:
        markup.append(f'<div class="aq-method"><div class="aq-method-kicker">{escape(kicker)}</div>'
                      f'<b>{escape(title)}</b><p>{escape(body)}</p>'
                      f'<div class="aq-method-value">{escape(value)}</div></div>')
    st.markdown("".join(markup) + "</div>", unsafe_allow_html=True)


def flow_grid(items):
    markup = ['<div class="aq-flow">']
    for kicker, title, body in items:
        markup.append(f'<div class="aq-flow-box"><small>{escape(kicker)}</small>'
                      f'<b>{escape(title)}</b><p>{escape(body)}</p></div>')
    st.markdown("".join(markup) + "</div>", unsafe_allow_html=True)


def show_altair(chart, height=275):
    styled = (chart.properties(height=height, background=PAPER)
              .configure_view(stroke=None)
              .configure_axis(labelFontSize=11, titleFontSize=11, labelColor="#587184",
                              titleColor=INK, gridColor=GRID_COLOR, tickColor=GRID_COLOR,
                              domainColor=GRID_COLOR, labelPadding=6, titlePadding=9)
              .configure_legend(orient="top", labelFontSize=11, titleFontSize=11,
                                labelColor=INK, titleColor=INK, symbolStrokeWidth=3, symbolOpacity=1)
              .configure_title(color=INK, fontSize=15, anchor="start"))
    st.altair_chart(styled, width="stretch", theme=None)


def footer():
    st.markdown('<div class="aq-footer"><span>Air Quality Lab · Kelompok 9 · Teknik Elektro ITK</span>'
                '<span>Forecasting CO +1 jam · Riset akademik, bukan ISPU resmi</span></div>',
                unsafe_allow_html=True)
