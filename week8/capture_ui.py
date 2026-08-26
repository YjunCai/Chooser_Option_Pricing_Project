"""
Capture Streamlit UI screenshots for the Week-8 final report & demo.
=====================================================================
Connects to the running app (http://localhost:8511), waits for full render,
and saves a screenshot of the pricing overview plus each of the six tabs.

Output -> assets/ui_*.png  (crisp, device_scale_factor=2)

Run:
    streamlit run streamlit_app.py          # terminal 1
    python capture_ui.py                     # terminal 2
"""

import time

from playwright.sync_api import sync_playwright

URL = 'http://localhost:8511'
OUT = 'assets'

TABS = {
    'trend': '📈 价格趋势',
    'sensitivity': '📊 敏感性分析',
    'scenarios': '🌪 极端场景',
    'metrics': '🏆 性能指标',
    'error_margins': '🔬 误差范围',
    'live_data': '📡 实时数据',
}


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1600, 'height': 950},
                                device_scale_factor=2)
        page.goto(URL, wait_until='networkidle')
        page.wait_for_selector('text=Chooser Option Pricing Tool', timeout=30000)
        page.wait_for_selector('text=BSM(vol_21d) 价格', timeout=30000)
        time.sleep(3)

        # 1. overview
        page.evaluate('window.scrollTo(0,0)')
        time.sleep(1)
        page.screenshot(path=f'{OUT}/ui_overview.png')
        print('saved ui_overview.png')

        # 2. each tab
        for key, label in TABS.items():
            try:
                page.get_by_role('tab', name=label).first.click()
                time.sleep(4.0)
                page.evaluate('window.scrollTo(0,0)')
                time.sleep(1.2)
                page.screenshot(path=f'{OUT}/ui_{key}.png')
                print(f'saved ui_{key}.png')
            except Exception as exc:
                print(f'tab {key} failed: {exc}')

        # 3. cropped sidebar shot
        page.evaluate('window.scrollTo(0,0)')
        time.sleep(1)
        sb = page.locator('[data-testid="stSidebar"]')
        try:
            sb.screenshot(path=f'{OUT}/ui_sidebar.png')
            print('saved ui_sidebar.png')
        except Exception as exc:
            print('sidebar shot failed:', exc)

        browser.close()


if __name__ == '__main__':
    main()
