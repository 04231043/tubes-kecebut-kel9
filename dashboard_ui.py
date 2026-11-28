"""Presentation components for the Air Quality Lab dashboard."""
from html import escape
from pathlib import Path

import streamlit as st

COLORS = ["#087a6b", "#3c62ac", "#ae6433", "#795399", "#657b8a"]
INK, GRID_COLOR, PAPER = "#162b3a", "#e4ecef", "#ffffff"
STYLE = Path(__file__).resolve().parent / "assets" / "dashboard.css"


def init_style():
    st.html(f"<style>{STYLE.read_text(encoding='utf-8')}</style>")


def sidebar_brand():
    st.sidebar.html("""
    <div class="aq-brand">
      <span class="aq-brand-icon" aria-hidden="true"><i></i><i></i><i></i></span>
      <div><div class="aq-brand-name">Air Quality Lab</div>
      <div class="aq-brand-context">Teknik Elektro ITK · Kelompok 9</div></div>
    </div>
    <div class="aq-side-label">Navigasi</div>
    """)


def sidebar_status(custom_run):
    source = "Eksperimen sesi" if custom_run else "UCI Air Quality · 2004–2005"
    st.sidebar.html(f"""
    <div class="aq-side-status">
      <span aria-hidden="true"></span><div>
      <b>Data historis</b>
      <p>{source}</p></div>
    </div>
    """)


def hero():
    st.html("""
    <section class="aq-hero" aria-labelledby="pantau-co">
      <div class="aq-hero-copy">
        <p class="aq-hero-label"><span></span> Riset kualitas udara</p>
        <h1 id="pantau-co">Pantau CO</h1>
        <p class="aq-hero-description">Prakiraan karbon monoksida satu jam ke depan dari riwayat sensor.</p>
        <p class="aq-hero-method">GA + PSO + RNN <span>·</span> Fuzzy Mamdani</p>
      </div>
      <div class="aq-hero-art" aria-hidden="true">
        <div class="aq-hero-tile"><span>+1 JAM</span><b>CO</b><small>mg/m³</small></div>
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


def page_header(section, title, description):
    label = f'<p class="aq-page-label">{escape(section)}</p>' if section != title else ""
    st.html(f'<div class="aq-page-intro">{label}<h1>{escape(title)}</h1><p>{escape(description)}</p></div>')


def show_altair(chart, height):
    styled = (chart.properties(height=height, background=PAPER)
              .configure_view(stroke=None)
              .configure_axis(labelFontSize=11, titleFontSize=11, labelColor="#405966",
                              titleColor=INK, gridColor=GRID_COLOR, tickColor=GRID_COLOR,
                              domainColor=GRID_COLOR, labelPadding=5, titlePadding=8)
              .configure_legend(orient="top", labelFontSize=11, labelColor=INK,
                                symbolStrokeWidth=3, symbolOpacity=1))
    st.altair_chart(styled, width="stretch", theme=None)


def footer():
    st.html("""
    <footer class="aq-footer">
      <span>Air Quality Lab · TE2514031 Kecerdasan Buatan</span>
      <a href="https://github.com/04231043/tubes-kecebut-kel9" target="_blank" rel="noopener noreferrer">Repositori GitHub ↗</a>
    </footer>
    """)
