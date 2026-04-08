import uuid
import streamlit as st
import requests
import os
import sys
import threading
from supabase import create_client, Client
from streamlit_javascript import st_javascript
from datetime import datetime, timezone
from google.colab import userdata

def get_supabase_client():
    try:
        url = userdata.get('SUPABASE_URL')
        key = userdata.get('SUPABASE_KEY')
        if not url or not key: return None
        return create_client(url, key)
    except: return None

def get_real_client_ip():
    if "cached_ip" in st.session_state:
        return st.session_state.cached_ip
    try:
        js_code = "await fetch('https://api.ipify.org?format=json').then(r => r.json()).then(d => d.ip)"
        client_ip = st_javascript(js_code, key="ip_tracker_js")
        if not client_ip or client_ip == 0: return None 
        st.session_state.cached_ip = client_ip
        return client_ip
    except: return "Unknown"

def get_or_create_session_id():
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = uuid.uuid4().hex
    return st.session_state['session_id']

def _async_log_worker(log_data, real_ip):
    try:
        loc_data = {}
        if real_ip and real_ip not in ["Unknown", None]:
            try:
                res = requests.get(f"http://ip-api.com/json/{real_ip}?fields=status,country,regionName,city,lat,lon", timeout=1)
                if res.status_code == 200:
                    loc_data = res.json()
            except:
                pass

        log_data.update({
            "country": loc_data.get('country', "Unknown"),
            "region": loc_data.get('regionName', "Unknown"),
            "city": loc_data.get('city', "Unknown"),
            "lat": loc_data.get('lat', 0.0),
            "lon": loc_data.get('lon', 0.0),
        })

        client = get_supabase_client()
        if client:
            client.table('usage_logs').insert(log_data).execute()
            
    except Exception as e:
        print(f"🚨 [Async Tracker Error]: {e}")

def log_app_usage(app_name="unknown_app", action="page_view", details=None):
    try:
        real_ip = get_real_client_ip()
        current_session = get_or_create_session_id()
        user_agent = st.context.headers.get("User-Agent", "Unknown") if hasattr(st, "context") else "Unknown"
        utc_time = datetime.now(timezone.utc).isoformat()

        if user_agent and any(keyword in user_agent.lower() for keyword in ["bot", "uptime", "cron"]):
            return False

        log_data = {
            "session_id": current_session,
            "app_name": app_name,
            "action": action,
            "timestamp": utc_time,
            "ip_address": real_ip if real_ip else "Unknown",
            "details": details if isinstance(details, dict) else {"info": str(details)},
            "user_agent": user_agent
        }

        threading.Thread(target=_async_log_worker, args=(log_data, real_ip), daemon=True).start()
        
        return True
    except Exception as e:
        print(f"🚨 트래커 시작 실패: {e}")
        return False
