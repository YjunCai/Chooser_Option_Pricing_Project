"""
Capture Streamlit UI screenshots for the Week-7 report.
=======================================================
Connects to the running app (http://localhost:8511), waits for full render,
and saves a screenshot of the pricing overview plus each of the four tabs.

Output -> assets/ui_*.png  (crisp, device_scale_factor=2)
"""

import time

from playwright.sync_api import sync_playwright

URL = 'http://localhost:8511'
OUT = 'assets'

TABS = {
    'sensitivity': '📊 敏感性分析',
    'scenarios': '🌪 极端场景',
    'error_margins': '🔬 误差范围',
    'live_data': '📡 实时数据',
}


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1500, 'height': 950},
                                device_scale_factor=2)
        page.goto(URL, wait_until='networkidle')
        # wait for the app to fully render (metric cards + title)
        page.wait_for_selector('text=Chooser Option Pricing Tool', timeout=30000)
        page.wait_for_selector('text=BSM(vol_21d) 价格', timeout=30000)
        time.sleep(3)

        # 1. overview: scroll to top, capture pricing + sidebar
        page.evaluate('window.scrollTo(0,0)')
        time.sleep(1)
        page.screenshot(path=f'{OUT}/ui_overview.png')
        print('saved ui_overview.png')

        # 2. each tab: click, wait, scroll top, screenshot
        for key, label in TABS.items():
            page.get_by_role('tab', name=label).first.click()
            time.sleep(3.5)
            page.evaluate('window.scrollTo(0,0)')
            time.sleep(1.0)
            page.screenshot(path=f'{OUT}/ui_{key}.png')
            print(f'saved ui_{key}.png')

        # 3. a cropped sidebar shot for the report sidebar figure
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
