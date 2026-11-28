"""Presentation components for the Air Quality Lab dashboard."""
from base64 import b64encode
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

COLORS = ["#5145b5", "#087b80", "#b65b25", "#a54a86", "#52667d"]
INK, GRID_COLOR, PAPER = "#192338", "#e2e7ef", "#ffffff"
ASSETS = Path(__file__).resolve().parent / "assets"


def init_style():
    st.html(f"<style>{(ASSETS / 'dashboard.css').read_text(encoding='utf-8')}</style>")


def sidebar_brand():
    st.sidebar.html("""
    <div class="aq-brand">
      <span class="aq-brand-icon" aria-hidden="true"><i></i><i></i><i></i></span>
      <div><div class="aq-brand-name">Air Quality Lab</div>
      <div class="aq-brand-context">Teknik Elektro ITK · Kelompok 9</div></div>
    </div>
    <div class="aq-side-label">Dashboard</div>
    """)


def sidebar_status(custom_run):
    source = "Eksperimen sesi" if custom_run else "UCI Air Quality · 2004–2005"
    st.sidebar.html(f"""
    <div class="aq-side-status">
      <b>Data historis</b>
      <p>{source}</p>
    </div>
    """)


@lru_cache(maxsize=1)
def hero_photo():
    return b64encode((ASSETS / "urban-avenue.webp").read_bytes()).decode("ascii")


def hero():
    photo = f'<img src="data:image/webp;base64,{hero_photo()}" alt="" aria-hidden="true">'
    st.html(f"""
    <section class="aq-hero" aria-labelledby="pantau-co">
      {photo}
      <div class="aq-hero-copy">
        <p class="aq-hero-eyebrow">Air Quality Lab · Kelompok 9</p>
        <h1 id="pantau-co">Pantau CO</h1>
        <p class="aq-hero-description">Prakiraan karbon monoksida satu jam ke depan.</p>
      </div>
    </section>
    """)


def overview_metrics(items):
    cards = (
        f'<div class="aq-stat"><dt>{escape(label)}</dt><dd>{escape(value)}'
        f'<span>{escape(detail)}</span></dd></div>'
        for label, value, detail in items
    )
    st.html('<dl class="aq-stat-grid">' + "".join(cards) + "</dl>")


def overview_results(items):
    results = (
        f'<div><dt>{escape(label)}</dt>'
        f'<dd><strong>{escape(value)}</strong> <small>{escape(unit)}</small>'
        f'<span>{escape(model)}</span></dd></div>'
        for label, value, unit, model in items
    )
    st.html('<dl class="aq-results">' + "".join(results) + "</dl>")


def model_architecture():
    st.html("""
    <div class="aq-architecture">
      <div class="aq-flow-row">
        <h4>Pelatihan</h4>
        <ol aria-label="Alur pelatihan">
          <li><b>Data sensor</b><span>Latih dan validasi</span></li>
          <li><b>Genetic Algorithm</b><span>Memilih kanal</span></li>
          <li><b>PSO</b><span>Mencari konfigurasi RNN</span></li>
          <li><b>Checkpoint RNN</b><span>Menyimpan bobot terlatih</span></li>
        </ol>
      </div>
      <div class="aq-flow-row">
        <h4>Prakiraan</h4>
        <ol aria-label="Alur prakiraan">
          <li><b>Riwayat sensor</b><span>Observasi sampai jam t</span></li>
          <li><b>RNN terlatih</b><span>Memuat checkpoint</span></li>
          <li><b>CO satu jam ke depan</b><span>Konsentrasi dalam mg/m³</span></li>
          <li><b>Fuzzy Mamdani</b><span>Prioritas verifikasi sensor</span></li>
        </ol>
      </div>
    </div>
    """)


def page_header(section, title, description):
    label = f'<p class="aq-page-label">{escape(section)}</p>' if section != title else ""
    st.html(f'<div class="aq-page-intro">{label}<h1>{escape(title)}</h1><p>{escape(description)}</p></div>')


def show_altair(chart, height):
    styled = (chart.properties(height=height, background=PAPER)
              .configure_view(stroke=None)
              .configure_axis(labelFontSize=11, titleFontSize=11, labelColor="#526078",
                              titleColor=INK, gridColor=GRID_COLOR, tickColor=GRID_COLOR,
                              domainColor=GRID_COLOR, labelPadding=6, titlePadding=10)
              .configure_legend(orient="top", labelFontSize=11, labelColor=INK,
                                symbolStrokeWidth=2.5, symbolOpacity=1, padding=3, offset=10))
    st.altair_chart(styled, width="stretch", theme=None)


def footer():
    st.html("""
    <footer class="aq-footer">
      <span>Air Quality Lab · TE2514031 Kecerdasan Buatan</span>
      <a href="https://github.com/04231043/tubes-kecebut-kel9" target="_blank" rel="noopener noreferrer">Repositori GitHub ↗</a>
    </footer>
    """)
