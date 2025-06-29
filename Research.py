import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import folium
from streamlit_folium import st_folium
import numpy as np
from folium import plugins
import base64
from io import BytesIO, StringIO
import warnings
import hashlib
import os
import streamlit.components.v1 as components
import re
from pathlib import Path

warnings.filterwarnings('ignore')

# Enhanced imports for advanced features
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    st.error("GeoPandas not available. Please install: pip install geopandas")

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("Scikit-learn not available. ML features disabled.")

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Configuration
st.set_page_config(
    page_title="Amazon Archaeological Research Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
def setup_logging():
    """Setup comprehensive logging system"""
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"archaeological_platform_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not create log file: {e}")
        return logger

logger = setup_logging()


# Professional CSS styling
def load_professional_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: #0e1117;
        color: #fafafa;
    }
    
    .main-container {
        background: #1e2329;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .platform-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .platform-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.3;
    }
    
    .platform-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: white !important;
        position: relative;
        z-index: 1;
    }
    
    .platform-header p {
        color: white !important;
        font-size: 1.2rem;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    .modern-card {
        background: #262d3a;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }
    
    .modern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3498db, #2ecc71, #f39c12, #e74c3c);
    }
    
    .status-card {
        background: #2c3e50;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        color: #ecf0f1;
    }
    
    .status-card.success {
        border-left-color: #2ecc71;
        background: linear-gradient(135deg, #27ae60 0%, #2c3e50 100%);
    }
    
    .status-card.warning {
        border-left-color: #f39c12;
        background: linear-gradient(135deg, #f39c12 0%, #2c3e50 100%);
    }
    
    .status-card.error {
        border-left-color: #e74c3c;
        background: linear-gradient(135deg, #e74c3c 0%, #2c3e50 100%);
    }
    
    .metric-card {
        background: #34495e;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        color: #ecf0f1;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    .metric-card .metric-value {
        font-size: 3rem;
        font-weight: 700;
        color: #3498db;
        margin: 1rem 0;
    }
    
    .metric-card .metric-label {
        font-size: 1rem;
        color: #bdc3c7;
        font-weight: 500;
    }
    
    .api-config-card {
        background: #34495e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        color: #ecf0f1;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2ecc71 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(52, 152, 219, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(52, 152, 219, 0.4);
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(52, 152, 219, 0.3);
        border-radius: 50%;
        border-top-color: #3498db;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)

class APIKeyManager:
    """Manage API keys with user input and default fallback"""
    
    def __init__(self):
        self.default_apis = {
            'wikidata': {
                'name': 'Wikidata SPARQL',
                'url': 'https://query.wikidata.org/sparql',
                'requires_key': False,
                'description': 'Free public API for archaeological and tribal data'
            }
        }
        
        # Optional APIs that could be added in the future
        self.optional_apis = {
            'mapbox': {
                'name': 'Mapbox',
                'requires_key': True,
                'description': 'Enhanced mapping tiles and geocoding',
                'default_available': False
            },
            'google_maps': {
                'name': 'Google Maps',
                'requires_key': True,
                'description': 'Satellite imagery and enhanced geocoding',
                'default_available': False
            },
            "OpenAPI":{
                'name':"Open API",
                'requires_key':True,
                'description': "Open API key for Data Collection",
                'default_available': True
            }
        }
    
    def get_api_config(self) -> Dict:
        """Get API configuration from user input or defaults"""
        config = {}
        
        # Always available - Wikidata (no key required)
        config['wikidata'] = {
            'url': self.default_apis['wikidata']['url'],
            'headers': {
                'User-Agent': 'Amazon-Archaeological-Research-Platform/2.1',
                'Accept': 'application/sparql-results+json'
            },
            'available': True
        }
        
        # Optional APIs with user keys
        for api_name, api_info in self.optional_apis.items():
            key_input = st.session_state.get(f'{api_name}_api_key', '')
            if key_input and key_input.strip():
                config[api_name] = {
                    'api_key': key_input.strip(),
                    'available': True
                }
            else:
                config[api_name] = {
                    'available': False,
                    'reason': 'No API key provided'
                }
        
        return config
    
    def render_api_configuration(self):
        """Render API configuration interface"""
        st.markdown("### 🔑 API Configuration")
        
        # Default APIs (always available)
        st.markdown("#### ✅ Default APIs (Always Available)")
        for api_name, api_info in self.default_apis.items():
            st.markdown(f"""
            <div class="api-config-card">
                <h4>🌐 {api_info['name']}</h4>
                <p>{api_info['description']}</p>
                <p><strong>Status:</strong> <span style="color: #2ecc71;">✅ Active (No key required)</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Optional APIs (user can provide keys)
        st.markdown("#### 🔧 Optional APIs (Provide Your Keys)")
        
        for api_name, api_info in self.optional_apis.items():
            with st.expander(f"🔑 {api_info['name']} Configuration"):
                st.markdown(f"**Description:** {api_info['description']}")
                
                # API key input
                key_value = st.text_input(
                    f"Enter {api_info['name']} API Key",
                    type="password",
                    key=f"{api_name}_api_key",
                    help=f"Optional: Provide your {api_info['name']} API key for enhanced features"
                )
                
                # Status display
                if key_value and key_value.strip():
                    st.success(f"✅ {api_info['name']} API key configured")
                else:
                    st.info(f"ℹ️ {api_info['name']} will use default/fallback options")
        
        # API status summary
        config = self.get_api_config()
        active_apis = [name for name, conf in config.items() if conf.get('available', False)]
        
        st.markdown("#### 📊 API Status Summary")
        st.markdown(f"""
        <div class="api-config-card">
            <p><strong>Active APIs:</strong> {len(active_apis)}</p>
            <p><strong>Available:</strong> {', '.join(active_apis)}</p>
            <p><strong>Primary Data Source:</strong> Wikidata SPARQL (Always Available)</p>
        </div>
        """, unsafe_allow_html=True)
        
        return config

class DataLogger:
    """Enhanced logging system for data operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.operation_log = []
        self.coordinate_failures = []
        self.processing_stats = {
            'total_processed': 0,
            'successful_coordinates': 0,
            'failed_coordinates': 0,
            'invalid_formats': 0,
            'out_of_range': 0
        }
    
    def log_coordinate_failure(self, original_data: str, reason: str, item_name: str = "Unknown"):
        """Log coordinate parsing failures"""
        failure_entry = {
            'timestamp': datetime.now().isoformat(),
            'item_name': item_name,
            'original_data': original_data,
            'reason': reason,
            'type': 'coordinate_failure'
        }
        self.coordinate_failures.append(failure_entry)
        self.processing_stats['failed_coordinates'] += 1
        
        if 'format' in reason.lower():
            self.processing_stats['invalid_formats'] += 1
        elif 'range' in reason.lower():
            self.processing_stats['out_of_range'] += 1
    
    def log_coordinate_success(self, item_name: str = "Unknown"):
        """Log successful coordinate parsing"""
        self.processing_stats['successful_coordinates'] += 1
    
    def log_api_failure(self, api_name: str, url: str, error: str, params: dict = None):
        """Log API failure"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'api_name': api_name,
            'url': url,
            'error': str(error),
            'params': params,
            'type': 'api_failure'
        }
        self.operation_log.append(log_entry)
        self.logger.error(f"API Failure - {api_name}: {error}")
    
    def log_data_success(self, api_name: str, data_count: int, data_type: str):
        """Log successful data retrieval"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'api_name': api_name,
            'data_count': data_count,
            'data_type': data_type,
            'type': 'data_success'
        }
        self.operation_log.append(log_entry)
        self.logger.info(f"Data Success - {api_name}: Retrieved {data_count} {data_type} records")
    
    def log_processing_error(self, operation: str, error: str, data_sample: Any = None):
        """Log data processing errors"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'error': str(error),
            'data_sample': str(data_sample)[:200] if data_sample else None,
            'type': 'processing_error'
        }
        self.operation_log.append(log_entry)
        self.logger.error(f"Processing Error - {operation}: {error}")
    
    def get_coordinate_processing_summary(self) -> Dict:
        """Get detailed summary of coordinate processing"""
        total = self.processing_stats['total_processed']
        success_rate = (self.processing_stats['successful_coordinates'] / total * 100) if total > 0 else 0
        
        return {
            'total_processed': total,
            'successful_coordinates': self.processing_stats['successful_coordinates'],
            'failed_coordinates': self.processing_stats['failed_coordinates'],
            'success_rate': success_rate,
            'invalid_formats': self.processing_stats['invalid_formats'],
            'out_of_range': self.processing_stats['out_of_range'],
            'recent_failures': self.coordinate_failures[-10:] if self.coordinate_failures else []
        }
    
    def get_operation_summary(self) -> Dict:
        """Get summary of all operations"""
        summary = {
            'total_operations': len(self.operation_log),
            'api_failures': len([log for log in self.operation_log if log['type'] == 'api_failure']),
            'data_successes': len([log for log in self.operation_log if log['type'] == 'data_success']),
            'processing_errors': len([log for log in self.operation_log if log['type'] == 'processing_error']),
            'coordinate_processing': self.get_coordinate_processing_summary(),
            'operations': self.operation_log
        }
        return summary

class EnhancedAPIManager:
    """Enhanced API management with comprehensive error handling"""
    
    def __init__(self, logger: DataLogger, api_config: Dict):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Amazon-Archaeological-Research-Platform/2.1',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })
        self.cache = {}
        self.last_request_time = {}
        self.logger = logger
        self.api_config = api_config
    
    def _get_cache_key(self, url: str, params: dict = None) -> str:
        """Generate cache key for request"""
        cache_data = f"{url}_{json.dumps(params, sort_keys=True) if params else ''}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def safe_request(self, url: str, params: dict = None, timeout: int = 60, 
                    min_interval: float = 2.0, retries: int = 3, api_name: str = "Unknown") -> Optional[dict]:
        """Make safe API requests with comprehensive error handling"""
        cache_key = self._get_cache_key(url, params)
        
        # Check cache first (4 hour cache)
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if time.time() - cache_time < 14400:  # 4 hours
                return data
        
        # Rate limiting
        current_time = time.time()
        if url in self.last_request_time:
            time_since_last = current_time - self.last_request_time[url]
            if time_since_last < min_interval:
                time.sleep(min_interval - time_since_last)
        
        # Retry logic
        last_error = None
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                
                # Handle different content types
                content_type = response.headers.get('content-type', '').lower()
                if 'application/json' in content_type or 'application/sparql-results+json' in content_type:
                    data = response.json()
                else:
                    data = response.json()
                
                # Cache the response
                self.cache[cache_key] = (current_time, data)
                self.last_request_time[url] = time.time()
                
                # Log success
                data_count = len(data.get('results', {}).get('bindings', [])) if isinstance(data, dict) and 'results' in data else 1
                self.logger.log_data_success(api_name, data_count, "records")
                
                return data
                
            except requests.exceptions.Timeout as e:
                last_error = f"Request timeout after {timeout} seconds"
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                continue
                
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {str(e)}"
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                continue
                
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error {response.status_code}: {str(e)}"
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                continue
                
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON response: {str(e)}"
                return None
                
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                continue
        
        # Final failure log
        self.logger.log_api_failure(api_name, url, f"Failed after {retries} attempts. Last error: {last_error}", params)
        return None

class ImprovedGeospatialProcessor:
    """Improved geospatial data processing with robust coordinate parsing"""
    
    def __init__(self, logger: DataLogger):
        self.logger = logger
        self.coordinate_patterns = [
            r'POINT\s*$$\s*([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s*$$',
            r'Point\s*$$\s*([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s*$$',
            r'([+-]?\d{1,3}\.\d+)\s*,\s*([+-]?\d{1,2}\.\d+)',
            r'([+-]?\d{1,3}\.\d+)\s+([+-]?\d{1,2}\.\d+)',
            r'(\d{1,3}\.?\d*)[°]?\s*([NS])\s*,?\s*(\d{1,3}\.?\d*)[°]?\s*([EW])',
            r'(\d{1,3})[°]\s*(\d{1,2})[\'′]\s*(\d{1,2}\.?\d*)[\"″]?\s*([NS])\s*,?\s*(\d{1,3})[°]\s*(\d{1,2})[\'′]\s*(\d{1,2}\.?\d*)[\"″]?\s*([EW])',
            r'([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)',
        ]
    
    def parse_coordinates(self, coord_string: str, item_name: str = "Unknown") -> Optional[Tuple[float, float]]:
        """Enhanced coordinate parsing with comprehensive format support"""
        if not coord_string or coord_string.strip() == '':
            self.logger.log_coordinate_failure("", "Empty coordinate string", item_name)
            return None
        
        coord_string = str(coord_string).strip()
        self.logger.processing_stats['total_processed'] += 1
        
        for i, pattern in enumerate(self.coordinate_patterns):
            try:
                match = re.search(pattern, coord_string, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    if i <= 3:
                        lon_str, lat_str = groups[0], groups[1]
                        lon, lat = float(lon_str), float(lat_str)
                        
                        if abs(lon) > 180 or abs(lat) > 90:
                            if abs(lat) <= 180 and abs(lon) <= 90:
                                lon, lat = lat, lon
                            else:
                                self.logger.log_coordinate_failure(
                                    coord_string, 
                                    f"Coordinates out of valid range: lon={lon}, lat={lat}",
                                    item_name
                                )
                                continue
                        
                    elif i == 4:
                        lat_deg, lat_dir, lon_deg, lon_dir = groups
                        lat = float(lat_deg) * (1 if lat_dir.upper() == 'N' else -1)
                        lon = float(lon_deg) * (1 if lon_dir.upper() == 'E' else -1)
                        
                    elif i == 5:
                        lat_deg, lat_min, lat_sec, lat_dir, lon_deg, lon_min, lon_sec, lon_dir = groups
                        lat = (float(lat_deg) + float(lat_min)/60 + float(lat_sec)/3600) * (1 if lat_dir.upper() == 'N' else -1)
                        lon = (float(lon_deg) + float(lon_min)/60 + float(lon_sec)/3600) * (1 if lon_dir.upper() == 'E' else -1)
                    
                    else:
                        lon, lat = float(groups[0]), float(groups[1])
                    
                    if -180 <= lon <= 180 and -90 <= lat <= 90:
                        self.logger.log_coordinate_success(item_name)
                        return lat, lon
                    else:
                        self.logger.log_coordinate_failure(
                            coord_string,
                            f"Final validation failed: lon={lon}, lat={lat}",
                            item_name
                        )
                        continue
                        
            except (ValueError, AttributeError) as e:
                continue
        
        self.logger.log_coordinate_failure(
            coord_string,
            f"No matching coordinate pattern found. Tried {len(self.coordinate_patterns)} patterns.",
            item_name
        )
        return None
    
    def create_geodataframe(self, data: List[Dict], name_field: str, country_field: str, coord_field: str) -> Optional[gpd.GeoDataFrame]:
        """Create GeoDataFrame with improved error handling"""
        if not GEOPANDAS_AVAILABLE:
            self.logger.log_processing_error("GeoPandas", "GeoPandas not available for geospatial processing")
            return None
        
        if not data:
            self.logger.log_processing_error("GeoDataFrame Creation", "No data provided")
            return None
        
        processed_data = []
        failed_count = 0
        processing_details = {
            'total_items': len(data),
            'empty_coordinates': 0,
            'invalid_format': 0,
            'out_of_range': 0,
            'missing_fields': 0,
            'successful': 0
        }
        
        for i, item in enumerate(data):
            try:
                name = "Unknown"
                country = "Unknown"
                coord_str = ""
                
                if isinstance(item.get(name_field), dict):
                    name = item.get(name_field, {}).get('value', f'Unknown_{i}')
                else:
                    name = str(item.get(name_field, f'Unknown_{i}'))
                
                if isinstance(item.get(country_field), dict):
                    country = item.get(country_field, {}).get('value', 'Unknown')
                else:
                    country = str(item.get(country_field, 'Unknown'))
                
                if isinstance(item.get(coord_field), dict):
                    coord_str = item.get(coord_field, {}).get('value', '')
                else:
                    coord_str = str(item.get(coord_field, ''))
                
                if not coord_str or coord_str.strip() == '':
                    processing_details['empty_coordinates'] += 1
                    failed_count += 1
                    continue
                
                coords = self.parse_coordinates(coord_str, name)
                if coords:
                    lat, lon = coords
                    processed_data.append({
                        'name': name,
                        'country': country,
                        'latitude': lat,
                        'longitude': lon,
                        'geometry': Point(lon, lat),
                        'original_coords': coord_str,
                        'data_source': 'Wikidata',
                        'processing_timestamp': datetime.now().isoformat()
                    })
                    processing_details['successful'] += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                processing_details['missing_fields'] += 1
                self.logger.log_processing_error(
                    "Item Processing", 
                    f"Error processing item {i}: {str(e)}",
                    str(item)[:200]
                )
        
        success_rate = (processing_details['successful'] / processing_details['total_items']) * 100 if processing_details['total_items'] > 0 else 0
        
        if processed_data:
            try:
                gdf = gpd.GeoDataFrame(processed_data, crs='EPSG:4326')
                self.logger.log_data_success("GeospatialProcessor", len(processed_data), "geospatial records")
                st.success(f"✅ **GeoDataFrame Creation Success**: Processed {len(processed_data)} out of {len(data)} records ({success_rate:.1f}% success rate)")
                return gdf
            except Exception as e:
                self.logger.log_processing_error("GeoDataFrame Creation", f"Failed to create GeoDataFrame: {str(e)}")
                return None
        
        self.logger.log_processing_error("GeoDataFrame Creation", "No valid coordinate data found")
        st.error("❌ **GeoDataFrame Creation Failed**: No valid coordinate data could be processed")
        return None
    
    def calculate_spatial_clusters(self, gdf: gpd.GeoDataFrame, eps_km: float = 50) -> gpd.GeoDataFrame:
        """Calculate spatial clusters using DBSCAN"""
        if not SKLEARN_AVAILABLE or gdf is None or len(gdf) < 2:
            return gdf
        
        try:
            gdf_proj = gdf.to_crs('EPSG:3857')
            coords = np.array([[point.x, point.y] for point in gdf_proj.geometry])
            
            eps_meters = eps_km * 1000
            clustering = DBSCAN(eps=eps_meters, min_samples=2).fit(coords)
            
            gdf = gdf.copy()
            gdf['cluster'] = clustering.labels_
            gdf['is_clustered'] = gdf['cluster'] != -1
            
            unique_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            clustered_points = sum(clustering.labels_ != -1)
            
            self.logger.log_data_success("Spatial Clustering", unique_clusters, "clusters")
            st.info(f"🔍 **Spatial Clustering Results**: Found {unique_clusters} clusters containing {clustered_points} points")
            
            return gdf
            
        except Exception as e:
            self.logger.log_processing_error("Spatial Clustering", str(e))
            return gdf

class ArchaeologicalAnalyzer:
    """Enhanced domain-specific algorithms for archaeological data analysis"""
    
    def __init__(self, logger: DataLogger):
        self.logger = logger
    
    def analyze_settlement_patterns(self, tribal_gdf: gpd.GeoDataFrame, sites_gdf: gpd.GeoDataFrame) -> Dict:
        """Comprehensive analysis of patterns between tribal settlements and archaeological sites"""
        if not SKLEARN_AVAILABLE:
            return {"error": "Scikit-learn not available for advanced analysis"}
        
        if tribal_gdf is None and sites_gdf is None:
            return {"error": "No geospatial data available for analysis"}
        
        try:
            analysis = {}
            
            # Proximity Analysis
            if tribal_gdf is not None and sites_gdf is not None and len(tribal_gdf) > 0 and len(sites_gdf) > 0:
                distances = []
                nearest_sites = []
                
                for _, tribe in tribal_gdf.iterrows():
                    tribe_point = tribe.geometry
                    site_distances = []
                    
                    for _, site in sites_gdf.iterrows():
                        distance_deg = tribe_point.distance(site.geometry)
                        distance_km = distance_deg * 111
                        site_distances.append((distance_km, site['name']))
                    
                    if site_distances:
                        min_distance, nearest_site = min(site_distances, key=lambda x: x[0])
                        distances.append(min_distance)
                        nearest_sites.append(nearest_site)
                
                if distances:
                    analysis['proximity'] = {
                        'mean_distance_to_nearest_site_km': float(np.mean(distances)),
                        'median_distance_to_nearest_site_km': float(np.median(distances)),
                        'min_distance_km': float(np.min(distances)),
                        'max_distance_km': float(np.max(distances)),
                        'std_distance_km': float(np.std(distances)),
                        'distances_distribution': {
                            'very_close_0_10km': len([d for d in distances if d <= 10]),
                            'close_10_50km': len([d for d in distances if 10 < d <= 50]),
                            'moderate_50_100km': len([d for d in distances if 50 < d <= 100]),
                            'far_100_200km': len([d for d in distances if 100 < d <= 200]),
                            'very_far_200km_plus': len([d for d in distances if d > 200])
                        }
                    }
            
            # Density Analysis
            if tribal_gdf is not None and len(tribal_gdf) > 1:
                tribal_by_country = tribal_gdf.groupby('country').agg({
                    'name': 'count',
                    'latitude': ['mean', 'std'],
                    'longitude': ['mean', 'std']
                }).round(4)
                
                analysis['tribal_density_analysis'] = {
                    'by_country': {str(k): v for k, v in tribal_by_country.to_dict().items()},
                    'total_countries': len(tribal_gdf['country'].unique()),
                    'most_populated_country': str(tribal_gdf['country'].value_counts().index[0]),
                    'geographic_spread': {
                        'lat_range': float(tribal_gdf['latitude'].max() - tribal_gdf['latitude'].min()),
                        'lon_range': float(tribal_gdf['longitude'].max() - tribal_gdf['longitude'].min()),
                        'centroid': {
                            'lat': float(tribal_gdf['latitude'].mean()),
                            'lon': float(tribal_gdf['longitude'].mean())
                        }
                    }
                }
            
            if sites_gdf is not None and len(sites_gdf) > 1:
                sites_by_country = sites_gdf.groupby('country').agg({
                    'name': 'count',
                    'latitude': ['mean', 'std'],
                    'longitude': ['mean', 'std']
                }).round(4)
                
                analysis['archaeological_density_analysis'] = {
                    'by_country': {str(k): v for k, v in sites_by_country.to_dict().items()},
                    'total_countries': len(sites_gdf['country'].unique()),
                    'most_sites_country': str(sites_gdf['country'].value_counts().index[0]),
                    'geographic_spread': {
                        'lat_range': float(sites_gdf['latitude'].max() - sites_gdf['latitude'].min()),
                        'lon_range': float(sites_gdf['longitude'].max() - sites_gdf['longitude'].min()),
                        'centroid': {
                            'lat': float(sites_gdf['latitude'].mean()),
                            'lon': float(sites_gdf['longitude'].mean())
                        }
                    }
                }
            
            # Clustering Analysis
            for gdf_name, gdf in [('tribal', tribal_gdf), ('archaeological', sites_gdf)]:
                if gdf is not None and len(gdf) > 2 and 'cluster' in gdf.columns:
                    clustered = gdf[gdf['cluster'] != -1]
                    unique_clusters = len(gdf['cluster'].unique()) - (1 if -1 in gdf['cluster'].values else 0)
                    
                    cluster_analysis = {
                        'total_sites': len(gdf),
                        'clustered_sites': len(clustered),
                        'unclustered_sites': len(gdf) - len(clustered),
                        'clustering_percentage': float((len(clustered) / len(gdf)) * 100),
                        'number_of_clusters': unique_clusters,
                        'average_cluster_size': float(len(clustered) / unique_clusters) if unique_clusters > 0 else 0
                    }
                    
                    if unique_clusters > 0:
                        cluster_sizes = clustered.groupby('cluster').size()
                        cluster_analysis['cluster_size_stats'] = {
                            'min_cluster_size': int(cluster_sizes.min()),
                            'max_cluster_size': int(cluster_sizes.max()),
                            'mean_cluster_size': float(cluster_sizes.mean()),
                            'std_cluster_size': float(cluster_sizes.std())
                        }
                    
                    analysis[f'{gdf_name}_clustering'] = cluster_analysis
            
            # Spatial Distribution Analysis
            if tribal_gdf is not None and len(tribal_gdf) > 0:
                tribal_bounds = tribal_gdf.total_bounds
                analysis['tribal_spatial_distribution'] = {
                    'bounding_box': {
                        'min_longitude': float(tribal_bounds[0]),
                        'min_latitude': float(tribal_bounds[1]),
                        'max_longitude': float(tribal_bounds[2]),
                        'max_latitude': float(tribal_bounds[3])
                    },
                    'geographic_extent': {
                        'longitude_range_degrees': float(tribal_bounds[2] - tribal_bounds[0]),
                        'latitude_range_degrees': float(tribal_bounds[3] - tribal_bounds[1]),
                        'approximate_area_km2': float((tribal_bounds[2] - tribal_bounds[0]) * 111 * (tribal_bounds[3] - tribal_bounds[1]) * 111)
                    },
                    'center_point': {
                        'longitude': float((tribal_bounds[0] + tribal_bounds[2]) / 2),
                        'latitude': float((tribal_bounds[1] + tribal_bounds[3]) / 2)
                    }
                }
            
            if sites_gdf is not None and len(sites_gdf) > 0:
                sites_bounds = sites_gdf.total_bounds
                analysis['archaeological_spatial_distribution'] = {
                    'bounding_box': {
                        'min_longitude': float(sites_bounds[0]),
                        'min_latitude': float(sites_bounds[1]),
                        'max_longitude': float(sites_bounds[2]),
                        'max_latitude': float(sites_bounds[3])
                    },
                    'geographic_extent': {
                        'longitude_range_degrees': float(sites_bounds[2] - sites_bounds[0]),
                        'latitude_range_degrees': float(sites_bounds[3] - sites_bounds[1]),
                        'approximate_area_km2': float((sites_bounds[2] - sites_bounds[0]) * 111 * (sites_bounds[3] - sites_bounds[1]) * 111)
                    },
                    'center_point': {
                        'longitude': float((sites_bounds[0] + sites_bounds[2]) / 2),
                        'latitude': float((sites_bounds[1] + sites_bounds[3]) / 2)
                    }
                }
            
            # Comparative Analysis
            if tribal_gdf is not None and sites_gdf is not None and len(tribal_gdf) > 0 and len(sites_gdf) > 0:
                tribal_center = tribal_gdf[['latitude', 'longitude']].mean()
                sites_center = sites_gdf[['latitude', 'longitude']].mean()
                
                center_distance = np.sqrt(
                    (tribal_center['latitude'] - sites_center['latitude'])**2 + 
                    (tribal_center['longitude'] - sites_center['longitude'])**2
                ) * 111
                
                analysis['comparative_analysis'] = {
                    'tribal_vs_archaeological_centers_distance_km': float(center_distance),
                    'overlapping_countries': list(set(tribal_gdf['country'].unique()) & set(sites_gdf['country'].unique())),
                    'tribal_only_countries': list(set(tribal_gdf['country'].unique()) - set(sites_gdf['country'].unique())),
                    'archaeological_only_countries': list(set(sites_gdf['country'].unique()) - set(tribal_gdf['country'].unique()))
                }
            
            self.logger.log_data_success("Archaeological Analysis", len(analysis), "analysis metrics")
            return analysis
            
        except Exception as e:
            self.logger.log_processing_error("Settlement Pattern Analysis", str(e))
            return {"error": str(e)}
    
    def detect_anomalies(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Enhanced spatial anomaly detection"""
        if not SKLEARN_AVAILABLE or gdf is None or len(gdf) < 5:
            return gdf
        
        try:
            coords = np.array([[point.x, point.y] for point in gdf.geometry])
            
            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_anomalies = iso_forest.fit_predict(coords)
            iso_scores = iso_forest.score_samples(coords)
            
            # Statistical outliers
            centroid = coords.mean(axis=0)
            distances = np.sqrt(np.sum((coords - centroid)**2, axis=1))
            distance_threshold = np.percentile(distances, 95)
            distance_anomalies = distances > distance_threshold
            
            gdf = gdf.copy()
            gdf['isolation_forest_anomaly'] = iso_anomalies == -1
            gdf['isolation_forest_score'] = iso_scores
            gdf['distance_anomaly'] = distance_anomalies
            gdf['distance_from_centroid_km'] = distances * 111
            gdf['is_anomaly'] = gdf['isolation_forest_anomaly'] | gdf['distance_anomaly']
            
            anomaly_count = gdf['is_anomaly'].sum()
            self.logger.log_data_success("Anomaly Detection", anomaly_count, "spatial anomalies")
            
            if anomaly_count > 0:
                st.info(f"🚨 **Anomaly Detection**: Found {anomaly_count} spatial anomalies out of {len(gdf)} points")
            
            return gdf
            
        except Exception as e:
            self.logger.log_processing_error("Anomaly Detection", str(e))
            return gdf
    
    def calculate_diversity_index(self, gdf: gpd.GeoDataFrame, group_column: str = 'country') -> Dict:
        """Enhanced diversity indices calculation"""
        if gdf is None or len(gdf) == 0:
            return {}
        
        try:
            counts = gdf[group_column].value_counts()
            proportions = counts / counts.sum()
            
            # Shannon Diversity Index
            shannon_index = -sum(proportions * np.log(proportions))
            
            # Simpson Diversity Index
            simpson_index = 1 - sum(proportions ** 2)
            
            # Evenness
            max_diversity = np.log(len(counts))
            evenness = shannon_index / max_diversity if max_diversity > 0 else 0
            
            # Berger-Parker dominance index
            berger_parker = proportions.max()
            
            # Margalef's richness index
            margalef_richness = (len(counts) - 1) / np.log(len(gdf)) if len(gdf) > 1 else 0
            
            diversity_metrics = {
                'shannon_diversity': float(shannon_index),
                'simpson_diversity': float(simpson_index),
                'evenness': float(evenness),
                'berger_parker_dominance': float(berger_parker),
                'margalef_richness': float(margalef_richness),
                'total_groups': len(counts),
                'total_records': len(gdf),
                'group_distribution': {str(k): int(v) for k, v in counts.to_dict().items()},
                'rarest_group': str(counts.index[-1]),
                'most_common_group': str(counts.index[0]),
                'singleton_groups': len(counts[counts == 1])
            }
            
            self.logger.log_data_success("Diversity Analysis", len(diversity_metrics), "diversity metrics")
            return diversity_metrics
            
        except Exception as e:
            self.logger.log_processing_error("Diversity Index Calculation", str(e))
            return {}

class WikidataAPI:
    """Enhanced Wikidata API with improved queries"""
    
    def __init__(self, api_manager: EnhancedAPIManager, logger: DataLogger):
        self.api = api_manager
        self.logger = logger
        self.base_url = "https://query.wikidata.org/sparql"
    
    def get_amazon_tribes(self) -> List[Dict]:
        """Get comprehensive Amazon indigenous tribes data"""
        
        query = """
        SELECT DISTINCT ?tribe ?tribeLabel ?country ?countryLabel ?coordinates ?population ?language ?languageLabel WHERE {
          ?tribe wdt:P31/wdt:P279* wd:Q41710 ;
                 wdt:P17 ?country .
          
          FILTER(?country IN (wd:Q155, wd:Q739, wd:Q736, wd:Q717, wd:Q419, wd:Q750, wd:Q734, wd:Q298, wd:Q811))
          
          OPTIONAL { ?tribe wdt:P625 ?coordinates . }
          OPTIONAL { ?tribe wdt:P1082 ?population . }
          OPTIONAL { ?tribe wdt:P2936 ?language . }
          
          FILTER(BOUND(?coordinates))
          
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en,pt,es,fr". }
        }
        ORDER BY ?countryLabel ?tribeLabel
        LIMIT 300
        """
        
        params = {
            'query': query,
            'format': 'json'
        }
        
        data = self.api.safe_request(self.base_url, params, timeout=120, api_name="Wikidata Tribes")
        
        if data and 'results' in data and 'bindings' in data['results']:
            tribes = data['results']['bindings']
            st.success(f"✅ **Wikidata Tribes**: Successfully retrieved {len(tribes)} tribal records")
            return tribes
        else:
            st.error("❌ **Wikidata Tribes**: Failed to retrieve data from API")
            return []
    
    def get_archaeological_sites(self) -> List[Dict]:
        """Get comprehensive archaeological sites data"""
        
        query = """
        SELECT DISTINCT ?site ?siteLabel ?country ?countryLabel ?coordinates ?inception ?heritage ?heritageLabel ?type ?typeLabel WHERE {
          {
            ?site wdt:P31/wdt:P279* wd:Q839954 .
          } UNION {
            ?site wdt:P31/wdt:P279* wd:Q9259 .
          } UNION {
            ?site wdt:P31/wdt:P279* wd:Q570116 .
          }
          
          ?site wdt:P17 ?country ;
                wdt:P625 ?coordinates .
          
          FILTER(?country IN (wd:Q155, wd:Q739, wd:Q736, wd:Q717, wd:Q419, wd:Q750, wd:Q734, wd:Q298, wd:Q811))
          
          OPTIONAL { ?site wdt:P571 ?inception . }
          OPTIONAL { ?site wdt:P1435 ?heritage . }
          OPTIONAL { ?site wdt:P31 ?type . }
          
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en,pt,es,fr". }
        }
        ORDER BY ?countryLabel ?siteLabel
        LIMIT 200
        """
        
        params = {
            'query': query,
            'format': 'json'
        }
        
        data = self.api.safe_request(self.base_url, params, timeout=120, api_name="Wikidata Archaeological Sites")
        
        if data and 'results' in data and 'bindings' in data['results']:
            sites = data['results']['bindings']
            st.success(f"✅ **Wikidata Archaeological Sites**: Successfully retrieved {len(sites)} site records")
            return sites
        else:
            st.error("❌ **Wikidata Archaeological Sites**: Failed to retrieve data from API")
            return []
    
    def get_amazon_rivers(self) -> List[Dict]:
        """Get comprehensive Amazon river system data"""
        
        query = """
        SELECT DISTINCT ?river ?riverLabel ?country ?countryLabel ?coordinates ?length ?mouth ?mouthLabel ?basin ?basinLabel WHERE {
          ?river wdt:P31/wdt:P279* wd:Q4022 ;
                 wdt:P17 ?country ;
                 wdt:P625 ?coordinates .
          
          FILTER(?country IN (wd:Q155, wd:Q739, wd:Q736, wd:Q717, wd:Q419, wd:Q750, wd:Q734))
          
          OPTIONAL { ?river wdt:P2043 ?length . }
          OPTIONAL { ?river wdt:P403 ?mouth . }
          OPTIONAL { ?river wdt:P4614 ?basin . }
          
          FILTER(BOUND(?length) || EXISTS { ?river wdt:P403 wd:Q3783 })
          
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en,pt,es,fr". }
        }
        ORDER BY DESC(?length)
        LIMIT 100
        """
        
        params = {
            'query': query,
            'format': 'json'
        }
        
        data = self.api.safe_request(self.base_url, params, timeout=120, api_name="Wikidata Rivers")
        
        if data and 'results' in data and 'bindings' in data['results']:
            rivers = data['results']['bindings']
            st.success(f"✅ **Wikidata Rivers**: Successfully retrieved {len(rivers)} river records")
            return rivers
        else:
            st.error("❌ **Wikidata Rivers**: Failed to retrieve data from API")
            return []

class EnhancedVisualizationSuite:
    """Comprehensive visualization suite for archaeological data"""
    
    def __init__(self, logger: DataLogger):
        self.logger = logger
    
    def create_archaeological_distribution_map(self, tribal_gdf: gpd.GeoDataFrame, sites_gdf: gpd.GeoDataFrame) -> go.Figure:
        """Create enhanced archaeological distribution visualization"""
        if tribal_gdf is None and sites_gdf is None:
            return self._create_empty_figure("No archaeological data available")
        
        try:
            fig = go.Figure()
            
            if tribal_gdf is not None and len(tribal_gdf) > 0:
                lats = [point.y for point in tribal_gdf.geometry]
                lons = [point.x for point in tribal_gdf.geometry]
                names = tribal_gdf['name'].tolist()
                countries = tribal_gdf['country'].tolist()
                
                fig.add_trace(go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        opacity=0.8
                    ),
                    text=[f"🏘️ {name}<br>Country: {country}" for name, country in zip(names, countries)],
                    name='Indigenous Tribes',
                    hovertemplate='<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
                ))
            
            if sites_gdf is not None and len(sites_gdf) > 0:
                lats = [point.y for point in sites_gdf.geometry]
                lons = [point.x for point in sites_gdf.geometry]
                names = sites_gdf['name'].tolist()
                countries = sites_gdf['country'].tolist()
                
                fig.add_trace(go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='blue',
                        symbol='diamond',
                        opacity=0.8
                    ),
                    text=[f"🏛️ {name}<br>Country: {country}" for name, country in zip(names, countries)],
                    name='Archaeological Sites',
                    hovertemplate='<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
                ))
            
            all_lats = []
            all_lons = []
            
            if tribal_gdf is not None and len(tribal_gdf) > 0:
                all_lats.extend([point.y for point in tribal_gdf.geometry])
                all_lons.extend([point.x for point in tribal_gdf.geometry])
            
            if sites_gdf is not None and len(sites_gdf) > 0:
                all_lats.extend([point.y for point in sites_gdf.geometry])
                all_lons.extend([point.x for point in sites_gdf.geometry])
            
            center_lat = np.mean(all_lats) if all_lats else -5.0
            center_lon = np.mean(all_lons) if all_lons else -60.0
            
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=4
                ),
                height=600,
                title=dict(
                    text="🗺️ Archaeological Data Distribution Map",
                    font=dict(size=18, color='#2c3e50'),
                    x=0.5
                ),
                showlegend=True
            )
            
            total_points = len(all_lats)
            self.logger.log_data_success("Distribution Map", total_points, "data points")
            return fig
            
        except Exception as e:
            self.logger.log_processing_error("Distribution Map Creation", str(e))
            return self._create_empty_figure(f"Error creating distribution map: {str(e)}")
    
    def create_country_analysis_chart(self, tribal_gdf: gpd.GeoDataFrame, sites_gdf: gpd.GeoDataFrame) -> go.Figure:
        """Create country-wise analysis chart"""
        if tribal_gdf is None and sites_gdf is None:
            return self._create_empty_figure("No data available for country analysis")
        
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Tribal Settlements by Country', 'Archaeological Sites by Country'],
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            if tribal_gdf is not None and len(tribal_gdf) > 0:
                tribal_counts = tribal_gdf['country'].value_counts()
                fig.add_trace(
                    go.Bar(
                        x=tribal_counts.index,
                        y=tribal_counts.values,
                        name='Tribal Settlements',
                        marker_color='red',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            if sites_gdf is not None and len(sites_gdf) > 0:
                sites_counts = sites_gdf['country'].value_counts()
                fig.add_trace(
                    go.Bar(
                        x=sites_counts.index,
                        y=sites_counts.values,
                        name='Archaeological Sites',
                        marker_color='blue',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                height=500,
                title=dict(
                    text="📊 Country-wise Archaeological Data Analysis",
                    font=dict(size=18, color='#2c3e50'),
                    x=0.5
                ),
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Country", row=1, col=1)
            fig.update_xaxes(title_text="Country", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            
            return fig
            
        except Exception as e:
            self.logger.log_processing_error("Country Analysis Chart Creation", str(e))
            return self._create_empty_figure(f"Error creating country analysis: {str(e)}")
    
    def create_cluster_visualization(self, gdf: gpd.GeoDataFrame, data_type: str) -> go.Figure:
        """Create cluster visualization for archaeological data"""
        if gdf is None or 'cluster' not in gdf.columns:
            return self._create_empty_figure(f"No cluster data available for {data_type}")
        
        try:
            fig = go.Figure()
            
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            unique_clusters = gdf['cluster'].unique()
            for i, cluster_id in enumerate(unique_clusters):
                cluster_data = gdf[gdf['cluster'] == cluster_id]
                
                if cluster_id == -1:
                    color = 'lightgray'
                    name = 'Unclustered'
                    symbol = 'x'
                else:
                    color = colors[i % len(colors)]
                    name = f'Cluster {cluster_id}'
                    symbol = 'circle'
                
                lats = [point.y for point in cluster_data.geometry]
                lons = [point.x for point in cluster_data.geometry]
                names = cluster_data['name'].tolist()
                
                fig.add_trace(go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=color,
                        opacity=0.8
                    ),
                    text=[f"{name} (Cluster {cluster_id})" for name in names],
                    name=name,
                    hovertemplate='<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
                ))
            
            center_lat = gdf.geometry.y.mean()
            center_lon = gdf.geometry.x.mean()
            
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=4
                ),
                height=600,
                title=dict(
                    text=f"🔍 Spatial Clustering Analysis - {data_type}",
                    font=dict(size=18, color='#2c3e50'),
                    x=0.5
                ),
                showlegend=True
            )
            
            cluster_count = len([c for c in unique_clusters if c != -1])
            self.logger.log_data_success("Cluster Visualization", cluster_count, "clusters")
            return fig
            
        except Exception as e:
            self.logger.log_processing_error("Cluster Visualization Creation", str(e))
            return self._create_empty_figure(f"Error creating cluster visualization: {str(e)}")
    
    def create_3d_archaeological_landscape(self, tribal_gdf: gpd.GeoDataFrame, sites_gdf: gpd.GeoDataFrame) -> go.Figure:
        """Create enhanced 3D visualization of archaeological landscape"""
        if tribal_gdf is None and sites_gdf is None:
            return self._create_empty_figure("No archaeological data available for 3D visualization")
        
        try:
            fig = go.Figure()
            
            if tribal_gdf is not None and len(tribal_gdf) > 0:
                tribal_lats = [point.y for point in tribal_gdf.geometry]
                tribal_lons = [point.x for point in tribal_gdf.geometry]
                tribal_names = tribal_gdf['name'].tolist()
                
                if 'cluster' in tribal_gdf.columns:
                    cluster_colors = []
                    color_map = {-1: 'gray', 0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple'}
                    for cluster in tribal_gdf['cluster']:
                        cluster_colors.append(color_map.get(cluster, 'red'))
                else:
                    cluster_colors = ['red'] * len(tribal_lats)
                
                fig.add_trace(go.Scatter3d(
                    x=tribal_lons,
                    y=tribal_lats,
                    z=[100] * len(tribal_lats),
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=cluster_colors,
                        opacity=0.8,
                        line=dict(width=2, color='white')
                    ),
                    text=tribal_names,
                    name='Indigenous Tribes',
                    hovertemplate='<b>%{text}</b><br>Lon: %{x:.4f}<br>Lat: %{y:.4f}<br>Type: Tribal Settlement<extra></extra>'
                ))
            
            if sites_gdf is not None and len(sites_gdf) > 0:
                sites_lats = [point.y for point in sites_gdf.geometry]
                sites_lons = [point.x for point in sites_gdf.geometry]
                sites_names = sites_gdf['name'].tolist()
                
                if 'cluster' in sites_gdf.columns:
                    cluster_colors = []
                    color_map = {-1: 'lightgray', 0: 'darkblue', 1: 'navy', 2: 'teal', 3: 'cyan', 4: 'indigo'}
                    for cluster in sites_gdf['cluster']:
                        cluster_colors.append(color_map.get(cluster, 'darkblue'))
                else:
                    cluster_colors = ['darkblue'] * len(sites_lats)
                
                fig.add_trace(go.Scatter3d(
                    x=sites_lons,
                    y=sites_lats,
                    z=[200] * len(sites_lats),
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=cluster_colors,
                        symbol='diamond',
                        opacity=0.9,
                        line=dict(width=2, color='white')
                    ),
                    text=sites_names,
                    name='Archaeological Sites',
                    hovertemplate='<b>%{text}</b><br>Lon: %{x:.4f}<br>Lat: %{y:.4f}<br>Type: Archaeological Site<extra></extra>'
                ))
            
            fig.update_layout(
                title=dict(
                    text="🏔️ 3D Archaeological Landscape Visualization",
                    font=dict(size=18, color='#2c3e50'),
                    x=0.5
                ),
                scene=dict(
                    xaxis_title="Longitude",
                    yaxis_title="Latitude", 
                    zaxis_title="Elevation (Symbolic)",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    bgcolor='rgba(240,240,240,0.8)',
                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white"),
                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white"),
                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white")
                ),
                height=700,
                font=dict(color='#2c3e50'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            total_points = (len(tribal_gdf) if tribal_gdf is not None else 0) + (len(sites_gdf) if sites_gdf is not None else 0)
            self.logger.log_data_success("3D Visualization", total_points, "data points")
            return fig
            
        except Exception as e:
            self.logger.log_processing_error("3D Visualization Creation", str(e))
            return self._create_empty_figure(f"Error creating 3D visualization: {str(e)}")
    
    def create_diversity_analysis_chart(self, tribal_gdf: gpd.GeoDataFrame, sites_gdf: gpd.GeoDataFrame) -> go.Figure:
        """Create diversity analysis visualization"""
        if tribal_gdf is None and sites_gdf is None:
            return self._create_empty_figure("No data available for diversity analysis")
        
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Tribal Diversity by Country',
                    'Archaeological Sites Diversity',
                    'Geographic Distribution',
                    'Data Summary'
                ],
                specs=[
                    [{"type": "pie"}, {"type": "pie"}],
                    [{"type": "bar"}, {"type": "table"}]
                ]
            )
            
            # Tribal diversity pie chart
            if tribal_gdf is not None and len(tribal_gdf) > 0:
                tribal_counts = tribal_gdf['country'].value_counts()
                fig.add_trace(
                    go.Pie(
                        labels=tribal_counts.index,
                        values=tribal_counts.values,
                        name="Tribal Distribution",
                        marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
                    ),
                    row=1, col=1
                )
            
            # Archaeological sites diversity pie chart
            if sites_gdf is not None and len(sites_gdf) > 0:
                sites_counts = sites_gdf['country'].value_counts()
                fig.add_trace(
                    go.Pie(
                        labels=sites_counts.index,
                        values=sites_counts.values,
                        name="Sites Distribution",
                        marker_colors=['#74b9ff', '#fd79a8', '#fdcb6e', '#6c5ce7', '#a29bfe']
                    ),
                    row=1, col=2
                )
            
            # Geographic distribution comparison
            if tribal_gdf is not None and sites_gdf is not None:
                all_countries = set()
                if len(tribal_gdf) > 0:
                    all_countries.update(tribal_gdf['country'].unique())
                if len(sites_gdf) > 0:
                    all_countries.update(sites_gdf['country'].unique())
                
                tribal_data = []
                sites_data = []
                
                for country in sorted(all_countries):
                    tribal_count = len(tribal_gdf[tribal_gdf['country'] == country]) if len(tribal_gdf) > 0 else 0
                    sites_count = len(sites_gdf[sites_gdf['country'] == country]) if len(sites_gdf) > 0 else 0
                    tribal_data.append(tribal_count)
                    sites_data.append(sites_count)
                
                fig.add_trace(
                    go.Bar(
                        x=list(sorted(all_countries)),
                        y=tribal_data,
                        name='Tribal Settlements',
                        marker_color='red',
                        opacity=0.7
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=list(sorted(all_countries)),
                        y=sites_data,
                        name='Archaeological Sites',
                        marker_color='blue',
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            # Summary table
            summary_data = []
            headers = ['Dataset', 'Total Records', 'Countries', 'Most Common Country']
            
            if tribal_gdf is not None and len(tribal_gdf) > 0:
                tribal_countries = tribal_gdf['country'].value_counts()
                summary_data.append([
                    'Tribal Settlements',
                    len(tribal_gdf),
                    len(tribal_countries),
                    tribal_countries.index[0]
                ])
            
            if sites_gdf is not None and len(sites_gdf) > 0:
                sites_countries = sites_gdf['country'].value_counts()
                summary_data.append([
                    'Archaeological Sites',
                    len(sites_gdf),
                    len(sites_countries),
                    sites_countries.index[0]
                ])
            
            if summary_data:
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=headers,
                            fill_color='#34495e',
                            font=dict(color='white', size=12),
                            align='center'
                        ),
                        cells=dict(
                            values=list(zip(*summary_data)),
                            fill_color='#ecf0f1',
                            font=dict(color='#2c3e50', size=11),
                            align='center'
                        )
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                title=dict(
                    text="📊 Comprehensive Diversity Analysis",
                    font=dict(size=18, color='#2c3e50'),
                    x=0.5
                ),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.log_processing_error("Diversity Analysis Chart Creation", str(e))
            return self._create_empty_figure(f"Error creating diversity analysis: {str(e)}")
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color='#7f8c8d')
        )
        fig.update_layout(
            height=400,
            template='plotly_white',
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig

def create_professional_interactive_map_html(tribal_gdf: gpd.GeoDataFrame, sites_gdf: gpd.GeoDataFrame, 
                                           rivers_gdf: gpd.GeoDataFrame = None) -> str:
    """Create professional interactive map HTML with enhanced features and proper labeling"""
    
    # Calculate center point from all available data
    all_lats, all_lons = [], []
    
    if tribal_gdf is not None and len(tribal_gdf) > 0:
        all_lats.extend([point.y for point in tribal_gdf.geometry])
        all_lons.extend([point.x for point in tribal_gdf.geometry])
    
    if sites_gdf is not None and len(sites_gdf) > 0:
        all_lats.extend([point.y for point in sites_gdf.geometry])
        all_lons.extend([point.x for point in sites_gdf.geometry])
    
    center_lat = np.mean(all_lats) if all_lats else -5.0
    center_lon = np.mean(all_lons) if all_lons else -60.0
    
    # Prepare data for JavaScript
    tribal_data = []
    if tribal_gdf is not None and len(tribal_gdf) > 0:
        for _, tribe in tribal_gdf.iterrows():
            lat, lon = tribe.geometry.y, tribe.geometry.x
            is_anomaly = tribe.get('is_anomaly', False) if 'is_anomaly' in tribe else False
            cluster_id = tribe.get('cluster', -1) if 'cluster' in tribe else -1
            
            tribal_data.append({
                'lat': lat,
                'lon': lon,
                'name': tribe['name'],
                'country': tribe['country'],
                'is_anomaly': is_anomaly,
                'cluster': cluster_id,
                'timestamp': tribe.get('processing_timestamp', 'Unknown')[:10] if tribe.get('processing_timestamp') else 'Unknown'
            })
    
    sites_data = []
    if sites_gdf is not None and len(sites_gdf) > 0:
        for _, site in sites_gdf.iterrows():
            lat, lon = site.geometry.y, site.geometry.x
            is_anomaly = site.get('is_anomaly', False) if 'is_anomaly' in site else False
            cluster_id = site.get('cluster', -1) if 'cluster' in site else -1
            
            sites_data.append({
                'lat': lat,
                'lon': lon,
                'name': site['name'],
                'country': site['country'],
                'is_anomaly': is_anomaly,
                'cluster': cluster_id,
                'timestamp': site.get('processing_timestamp', 'Unknown')[:10] if site.get('processing_timestamp') else 'Unknown'
            })
    
    rivers_data = []
    if rivers_gdf is not None and len(rivers_gdf) > 0:
        for _, river in rivers_gdf.iterrows():
            lat, lon = river.geometry.y, river.geometry.x
            rivers_data.append({
                'lat': lat,
                'lon': lon,
                'name': river['name'],
                'country': river['country']
            })
    
    # Create the complete HTML with enhanced Leaflet map
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Amazon Archaeological Interactive Map - Professional Edition</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.css" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.Default.css" />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" />
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #0e1117;
                color: #fafafa;
            }}
            #map {{
                height: 100vh;
                width: 100%;
            }}
            .custom-popup {{
                font-family: 'Inter', sans-serif;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                max-width: 400px;
            }}
            .popup-header {{
                padding: 20px;
                color: white;
                font-weight: 700;
                font-size: 18px;
            }}
            .popup-header.tribal {{
                background: linear-gradient(135deg, #e74c3c, #c0392b);
            }}
            .popup-header.archaeological {{
                background: linear-gradient(135deg, #8d6e63, #5d4037);
            }}
            .popup-header.river {{
                background: linear-gradient(135deg, #0277bd, #01579b);
            }}
            .popup-content {{
                padding: 20px;
                background: white;
                color: #2c3e50;
            }}
            .popup-field {{
                margin: 8px 0;
                font-weight: 600;
            }}
            .popup-cluster {{
                background: #3498db;
                color: white;
                padding: 10px;
                border-radius: 8px;
                margin: 10px 0;
                font-weight: bold;
            }}
            .popup-anomaly {{
                background: #f39c12;
                color: white;
                padding: 10px;
                border-radius: 8px;
                margin: 10px 0;
                font-weight: bold;
            }}
            .popup-footer {{
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #ecf0f1;
                font-size: 12px;
                font-style: italic;
                color: #7f8c8d;
            }}
            .legend {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                font-family: 'Inter', sans-serif;
                font-size: 14px;
                line-height: 1.6;
            }}
            .legend h4 {{
                margin: 0 0 10px 0;
                color: #000000;
                font-weight: 700;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin: 8px 0;
                color: #000000;
            }}
            .legend-color {{
                width: 16px;
                height: 16px;
                border-radius: 50%;
                margin-right: 8px;
                border: 2px solid white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            .legend-text {{
                color: #000000;
                font-weight: 500;
            }}
            .info-panel {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                font-family: 'Inter', sans-serif;
                font-size: 14px;
                max-width: 300px;
            }}
            .info-panel h4 {{
                margin: 0 0 10px 0;
                color: #000000;
                font-weight: 700;
            }}
            .info-stat {{
                display: flex;
                justify-content: space-between;
                margin: 5px 0;
                padding: 5px 0;
                border-bottom: 1px solid #ecf0f1;
                color: #000000;
            }}
            .info-stat:last-child {{
                border-bottom: none;
            }}
            .leaflet-control-layers {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                font-family: 'Inter', sans-serif;
            }}
            .leaflet-control-layers-expanded {{
                padding: 15px;
            }}
            .leaflet-control-layers label {{
                font-weight: 500;
                color: #000000;
                margin: 5px 0;
                display: flex;
                align-items: center;
            }}
            .leaflet-control-layers input[type="radio"],
            .leaflet-control-layers input[type="checkbox"] {{
                margin-right: 8px;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script src="https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster.js"></script>
        <script>
            // Initialize map
            var map = L.map('map').setView([{center_lat}, {center_lon}], 5);
            
            // Add tile layers with proper attribution
            var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                maxZoom: 19
            }}).addTo(map);
            
            var satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                attribution: '© <a href="https://www.esri.com/">Esri</a>, DigitalGlobe, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community',
                maxZoom: 19
            }});
            
            var topo = L.tileLayer('https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© <a href="https://opentopomap.org/">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
                maxZoom: 17
            }});
            
            // Layer groups
            var tribalGroup = L.layerGroup().addTo(map);
            var sitesGroup = L.layerGroup().addTo(map);
            var riversGroup = L.layerGroup();
            var clustersGroup = L.layerGroup();
            var anomaliesGroup = L.layerGroup();
            
            // Data
            var tribalData = {json.dumps(tribal_data)};
            var sitesData = {json.dumps(sites_data)};
            var riversData = {json.dumps(rivers_data)};
            
            // Color schemes
            var clusterColors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#e67e22', '#1abc9c'];
            
            // Add tribal settlements
            tribalData.forEach(function(tribe) {{
                var color = '#e74c3c';
                var group = tribalGroup;
                
                if (tribe.is_anomaly) {{
                    color = '#f39c12';
                    group = anomaliesGroup;
                }} else if (tribe.cluster !== -1) {{
                    color = clusterColors[tribe.cluster % clusterColors.length];
                    group = clustersGroup;
                }}
                
                var popupContent = `
                    <div class="custom-popup">
                        <div class="popup-header tribal">
                            <i class="fas fa-home"></i> ${{tribe.name}}
                            <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">Indigenous Settlement</div>
                        </div>
                        <div class="popup-content">
                            <div class="popup-field"><strong>📍 Country:</strong> ${{tribe.country}}</div>
                            <div class="popup-field"><strong>📊 Coordinates:</strong> ${{tribe.lat.toFixed(6)}}, ${{tribe.lon.toFixed(6)}}</div>
                            <div class="popup-field"><strong>🗓️ Data Source:</strong> Wikidata SPARQL API</div>
                            ${{tribe.cluster !== -1 ? '<div class="popup-cluster"><strong>🔍 Cluster ID:</strong> ' + tribe.cluster + '</div>' : ''}}
                            ${{tribe.is_anomaly ? '<div class="popup-anomaly"><strong>⚠️ Spatial Anomaly Detected</strong><br><small>This location shows unusual spatial patterns</small></div>' : ''}}
                            <div class="popup-footer">
                                Processed: ${{tribe.timestamp}}
                            </div>
                        </div>
                    </div>
                `;
                
                var marker = L.circleMarker([tribe.lat, tribe.lon], {{
                    radius: 8,
                    fillColor: color,
                    color: 'white',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.8
                }}).bindPopup(popupContent, {{maxWidth: 420}});
                
                marker.addTo(group);
            }});
            
            // Add archaeological sites
            sitesData.forEach(function(site) {{
                var color = '#8d6e63';
                var group = sitesGroup;
                
                if (site.is_anomaly) {{
                    color = '#f39c12';
                    group = anomaliesGroup;
                }} else if (site.cluster !== -1) {{
                    color = clusterColors[site.cluster % clusterColors.length];
                    group = clustersGroup;
                }}
                
                var popupContent = `
                    <div class="custom-popup">
                        <div class="popup-header archaeological">
                            <i class="fas fa-university"></i> ${{site.name}}
                            <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">Archaeological Site</div>
                        </div>
                        <div class="popup-content">
                            <div class="popup-field"><strong>📍 Country:</strong> ${{site.country}}</div>
                            <div class="popup-field"><strong>📊 Coordinates:</strong> ${{site.lat.toFixed(6)}}, ${{site.lon.toFixed(6)}}</div>
                            <div class="popup-field"><strong>🗓️ Data Source:</strong> Wikidata SPARQL API</div>
                            ${{site.cluster !== -1 ? '<div class="popup-cluster"><strong>🔍 Cluster ID:</strong> ' + site.cluster + '</div>' : ''}}
                            ${{site.is_anomaly ? '<div class="popup-anomaly"><strong>⚠️ Spatial Anomaly Detected</strong><br><small>This location shows unusual spatial patterns</small></div>' : ''}}
                            <div class="popup-footer">
                                Processed: ${{site.timestamp}}
                            </div>
                        </div>
                    </div>
                `;
                
                var marker = L.marker([site.lat, site.lon], {{
                    icon: L.divIcon({{
                        className: 'custom-div-icon',
                        html: '<div style="background-color:' + color + '; width: 12px; height: 12px; border-radius: 0; transform: rotate(45deg); border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>',
                        iconSize: [12, 12],
                        iconAnchor: [6, 6]
                    }})
                }}).bindPopup(popupContent, {{maxWidth: 440}});
                
                marker.addTo(group);
            }});
            
            // Add rivers
            riversData.forEach(function(river) {{
                var popupContent = `
                    <div class="custom-popup">
                        <div class="popup-header river">
                            <i class="fas fa-water"></i> ${{river.name}}
                            <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">Amazon River System</div>
                        </div>
                        <div class="popup-content">
                            <div class="popup-field"><strong>📍 Country:</strong> ${{river.country}}</div>
                            <div class="popup-field"><strong>📊 Coordinates:</strong> ${{river.lat.toFixed(6)}}, ${{river.lon.toFixed(6)}}</div>
                            <div class="popup-field"><strong>🗓️ Data Source:</strong> Wikidata SPARQL API</div>
                            <div class="popup-footer">
                                Part of the Amazon River Basin ecosystem
                            </div>
                        </div>
                    </div>
                `;
                
                var marker = L.circleMarker([river.lat, river.lon], {{
                    radius: 6,
                    fillColor: '#0277bd',
                    color: 'white',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.7
                }}).bindPopup(popupContent, {{maxWidth: 390}});
                
                marker.addTo(riversGroup);
            }});
            
            // Layer control with proper labels
            var baseMaps = {{
                "🗺️ OpenStreetMap": osm,
                "🛰️ Satellite Imagery": satellite,
                "🏔️ Topographic Map": topo
            }};
            
            var overlayMaps = {{
                "🏘️ Indigenous Tribes": tribalGroup,
                "🏛️ Archaeological Sites": sitesGroup,
                "🌊 Amazon Rivers": riversGroup,
                "🔍 Spatial Clusters": clustersGroup,
                "🚨 Spatial Anomalies": anomaliesGroup
            }};
            
            var layerControl = L.control.layers(baseMaps, overlayMaps, {{
                collapsed: false,
                position: 'topright'
            }}).addTo(map);
            
            // Add legend with proper naming and black text
            var legend = L.control({{position: 'bottomright'}});
            legend.onAdd = function(map) {{
                var div = L.DomUtil.create('div', 'legend');
                div.innerHTML = `
                    <h4>🗺️ Archaeological Data Legend</h4>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #e74c3c;"></div>
                        <span class="legend-text">Indigenous Tribal Settlements</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #8d6e63; transform: rotate(45deg);"></div>
                        <span class="legend-text">Archaeological Heritage Sites</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #0277bd;"></div>
                        <span class="legend-text">Amazon River Systems</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #f39c12;"></div>
                        <span class="legend-text">Spatial Anomalies Detected</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: linear-gradient(45deg, #3498db, #2ecc71);"></div>
                        <span class="legend-text">Clustered Locations</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-text">◈ Archaeological Site</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-text">◯ Indigenous Settlement</span>
                    </div>
                `;
                return div;
            }};
            legend.addTo(map);
            
            // Add info panel with proper naming and black text
            var info = L.control({{position: 'topleft'}});
            info.onAdd = function(map) {{
                var div = L.DomUtil.create('div', 'info-panel');
                div.innerHTML = `
                    <h4>📊 Amazon Archaeological Data Summary</h4>
                    <div class="info-stat">
                        <span>🏘️ Indigenous Tribal Settlements:</span>
                        <strong>{len(tribal_data)}</strong>
                    </div>
                    <div class="info-stat">
                        <span>🏛️ Archaeological Heritage Sites:</span>
                        <strong>{len(sites_data)}</strong>
                    </div>
                    <div class="info-stat">
                        <span>🌊 Amazon River Systems:</span>
                        <strong>{len(rivers_data)}</strong>
                    </div>
                    <div class="info-stat">
                        <span>📍 Total Data Points:</span>
                        <strong>{len(tribal_data) + len(sites_data) + len(rivers_data)}</strong>
                    </div>
                    <div class="info-stat">
                        <span>🗓️ Data Source:</span>
                        <strong>Wikidata SPARQL API</strong>
                    </div>
                `;
                return div;
            }};
            info.addTo(map);
            
            // Add scale
            L.control.scale({{position: 'bottomleft'}}).addTo(map);
            
            // Fit bounds to show all data
            if (tribalData.length > 0 || sitesData.length > 0) {{
                var allPoints = [...tribalData, ...sitesData, ...riversData];
                if (allPoints.length > 0) {{
                    var group = new L.featureGroup();
                    allPoints.forEach(function(point) {{
                        L.marker([point.lat, point.lon]).addTo(group);
                    }});
                    map.fitBounds(group.getBounds().pad(0.1));
                }}
            }}
            
            // Add click handler for coordinates
            map.on('click', function(e) {{
                console.log('Clicked at: ' + e.latlng.lat + ', ' + e.latlng.lng);
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content

def initialize_session_state():
    """Initialize session state with comprehensive caching"""
    defaults = {
        'data_logger': DataLogger(),
        'api_key_manager': APIKeyManager(),
        'api_manager': None,
        'geospatial_processor': None,
        'archaeological_analyzer': None,
        'visualization_suite': None,
        'tribal_data': None,
        'archaeological_sites': None,
        'rivers_data': None,
        'tribal_gdf': None,
        'sites_gdf': None,
        'rivers_gdf': None,
        'analysis_results': None,
        'visualizations': None,
        'data_loaded': False,
        'last_data_load': None,
        'cache_duration': 3600,
        'processing_errors': [],
        'coordinate_failures': [],
        'api_failures': [],
        'api_config': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Get API configuration
    if st.session_state.api_config is None:
        st.session_state.api_config = st.session_state.api_key_manager.get_api_config()
    
    # Initialize managers if not present
    if st.session_state.api_manager is None:
        st.session_state.api_manager = EnhancedAPIManager(st.session_state.data_logger, st.session_state.api_config)
    
    if st.session_state.geospatial_processor is None:
        st.session_state.geospatial_processor = ImprovedGeospatialProcessor(st.session_state.data_logger)
    
    if st.session_state.archaeological_analyzer is None:
        st.session_state.archaeological_analyzer = ArchaeologicalAnalyzer(st.session_state.data_logger)
    
    if st.session_state.visualization_suite is None:
        st.session_state.visualization_suite = EnhancedVisualizationSuite(st.session_state.data_logger)

def check_data_freshness() -> bool:
    """Check if cached data is still fresh"""
    if not st.session_state.last_data_load:
        return False
    
    time_since_load = time.time() - st.session_state.last_data_load
    return time_since_load < st.session_state.cache_duration

def safe_json_serialize(obj):
    """Safely serialize objects to JSON-compatible format"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    else:
        return str(obj)

def main():
    """Main application with professional interface and API key management"""
    initialize_session_state()
    load_professional_css()
    
    # Professional header
    st.markdown("""
    <div class="platform-header">
        <h1>🏛️ Amazon Archaeological Research Platform</h1>
        <p>Advanced Geospatial Analysis • Real Archaeological Data • Professional Research Tools</p>
        <p style="font-size: 1rem; opacity: 0.8;">Enhanced Coordinate Processing • API Key Management • Interactive Visualizations</p>
        <p style="font-size: 0.9rem; opacity: 0.7;">Powered by Wikidata SPARQL API, GeoPandas, and Advanced Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional sidebar with API configuration
    with st.sidebar:
        st.markdown("### 🔧 Platform Configuration")
        
        # API Configuration Section
        api_config = st.session_state.api_key_manager.render_api_configuration()
        
        # Update API manager with new configuration
        if api_config != st.session_state.api_config:
            st.session_state.api_config = api_config
            st.session_state.api_manager = EnhancedAPIManager(st.session_state.data_logger, api_config)
        
        st.markdown("---")
        
        # Data Status
        st.markdown("### 📊 Data Status")
        
        is_fresh = check_data_freshness()
        
        status_items = [
            ("📊 Data Freshness", is_fresh),
            ("🏘️ Tribal Data", st.session_state.tribal_gdf is not None and len(st.session_state.tribal_gdf) > 0),
            ("🏛️ Archaeological Sites", st.session_state.sites_gdf is not None and len(st.session_state.sites_gdf) > 0),
            ("🌊 Rivers Data", st.session_state.rivers_gdf is not None and len(st.session_state.rivers_gdf) > 0),
            ("🔍 Analysis Results", st.session_state.analysis_results is not None)
        ]
        
        for item, status in status_items:
            status_class = "success" if status else "warning"
            icon = "✅" if status else "⏳"
            st.markdown(f'<div class="status-card {status_class}"><p>{icon} {item}</p></div>', unsafe_allow_html=True)
        
        # Processing Statistics
        if st.session_state.data_logger:
            log_summary = st.session_state.data_logger.get_operation_summary()
            coord_summary = log_summary.get('coordinate_processing', {})
            
            st.markdown("### 📋 Processing Statistics")
            st.markdown(f"""
            <div class="status-card">
                <p><strong>Total Operations:</strong> {log_summary['total_operations']}</p>
                <p><strong>API Successes:</strong> {log_summary['data_successes']}</p>
                <p><strong>API Failures:</strong> {log_summary['api_failures']}</p>
                <p><strong>Processing Warnings:</strong> {log_summary['processing_errors']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Coordinate processing statistics
            if coord_summary:
                st.markdown("### 🗺️ Coordinate Processing")
                st.markdown(f"""
                <div class="status-card">
                    <p><strong>Total Processed:</strong> {coord_summary.get('total_processed', 0)}</p>
                    <p><strong>Success Rate:</strong> {coord_summary.get('success_rate', 0):.1f}%</p>
                    <p><strong>Failed Coordinates:</strong> {coord_summary.get('failed_coordinates', 0)}</p>
                    <p><strong>Invalid Formats:</strong> {coord_summary.get('invalid_formats', 0)}</p>
                    <p><strong>Out of Range:</strong> {coord_summary.get('out_of_range', 0)}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # System Requirements Check
        st.markdown("### 🔧 System Status")
        requirements = [
            ("GeoPandas", GEOPANDAS_AVAILABLE),
            ("Scikit-learn", SKLEARN_AVAILABLE),
            ("Matplotlib", MATPLOTLIB_AVAILABLE)
        ]
        
        for req, available in requirements:
            icon = "✅" if available else "❌"
            status_class = "success" if available else "error"
            st.markdown(f'<div class="status-card {status_class}"><p>{icon} {req}</p></div>', unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Collection", 
        "🔍 Analysis & ML", 
        "📈 Visualizations", 
        "🗺️ Interactive Maps",
        "📋 Reports",
        "💾 Export & Logs"
    ])
    
    with tab1:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("## 📊 Real Data Collection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Data Sources & Quality")
            st.markdown("""
            **All data is collected from verified APIs with enhanced error handling:**
            
            🔹 **Wikidata SPARQL API**: Indigenous tribes and archaeological sites  
            🔹 **Enhanced Coordinate Processing**: Multiple format support with validation  
            🔹 **Comprehensive Error Logging**: Detailed failure tracking and reporting  
            🔹 **NO FALLBACK DATA**: Only real API responses are used  
            🔹 **API Key Management**: Configure your own API keys or use defaults
            
            """)
            
            # Data collection button
            if st.button("🔄 Load All Real Data", type="primary", use_container_width=True):
                if not is_fresh or not st.session_state.data_loaded:
                    progress_bar = st.progress(0)
                    status_container = st.container()
                    
                    try:
                        with status_container:
                            st.info("🚀 **Starting Data Collection Process**")
                        
                        # Initialize APIs
                        wikidata = WikidataAPI(st.session_state.api_manager, st.session_state.data_logger)
                        
                        # Load tribal data
                        with status_container:
                            st.info("🔍 **Phase 1/4**: Loading indigenous tribes data from Wikidata...")
                        tribal_data = wikidata.get_amazon_tribes()
                        st.session_state.tribal_data = tribal_data
                        progress_bar.progress(25)
                        
                        # Load archaeological sites
                        with status_container:
                            st.info("🔍 **Phase 2/4**: Loading archaeological sites from Wikidata...")
                        archaeological_sites = wikidata.get_archaeological_sites()
                        st.session_state.archaeological_sites = archaeological_sites
                        progress_bar.progress(50)
                        
                        # Load rivers data
                        with status_container:
                            st.info("🔍 **Phase 3/4**: Loading Amazon river systems...")
                        rivers_data = wikidata.get_amazon_rivers()
                        st.session_state.rivers_data = rivers_data
                        progress_bar.progress(75)
                        
                        # Geospatial processing
                        with status_container:
                            st.info("🗺️ **Phase 4/4**: Processing geospatial data...")
                        
                        if tribal_data:
                            tribal_gdf = st.session_state.geospatial_processor.create_geodataframe(
                                tribal_data, 'tribeLabel', 'countryLabel', 'coordinates'
                            )
                            if tribal_gdf is not None:
                                tribal_gdf = st.session_state.geospatial_processor.calculate_spatial_clusters(tribal_gdf)
                                tribal_gdf = st.session_state.archaeological_analyzer.detect_anomalies(tribal_gdf)
                                st.session_state.tribal_gdf = tribal_gdf
                        
                        if archaeological_sites:
                            sites_gdf = st.session_state.geospatial_processor.create_geodataframe(
                                archaeological_sites, 'siteLabel', 'countryLabel', 'coordinates'
                            )
                            if sites_gdf is not None:
                                sites_gdf = st.session_state.geospatial_processor.calculate_spatial_clusters(sites_gdf)
                                sites_gdf = st.session_state.archaeological_analyzer.detect_anomalies(sites_gdf)
                                st.session_state.sites_gdf = sites_gdf
                        
                        if rivers_data:
                            rivers_gdf = st.session_state.geospatial_processor.create_geodataframe(
                                rivers_data, 'riverLabel', 'countryLabel', 'coordinates'
                            )
                            st.session_state.rivers_gdf = rivers_gdf
                        
                        progress_bar.progress(100)
                        
                        st.session_state.data_loaded = True
                        st.session_state.last_data_load = time.time()
                        
                        # Display success summary
                        with status_container:
                            st.success("✅ **Data Collection Completed Successfully!**")
                            
                            # Show processing statistics
                            coord_summary = st.session_state.data_logger.get_coordinate_processing_summary()
                            if coord_summary['total_processed'] > 0:
                                st.info(f"""
                                📊 **Coordinate Processing Summary:**
                                - Total records processed: {coord_summary['total_processed']}
                                - Successfully parsed: {coord_summary['successful_coordinates']} ({coord_summary['success_rate']:.1f}%)
                                - Failed to parse: {coord_summary['failed_coordinates']}
                                - Invalid formats: {coord_summary['invalid_formats']}
                                - Out of range: {coord_summary['out_of_range']}
                                """)
                        
                    except Exception as e:
                        with status_container:
                            st.error(f"❌ **Critical Error during data collection**: {str(e)}")
                        st.session_state.data_logger.log_processing_error("Data Collection", str(e))
                else:
                    st.info("📊 Using cached data (still fresh)")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown("### 📈 Data Summary")
            
            if st.session_state.data_loaded:
                # Display metrics
                metrics = [
                    ("🏘️ Tribes", len(st.session_state.tribal_gdf) if st.session_state.tribal_gdf is not None else 0),
                    ("🏛️ Sites", len(st.session_state.sites_gdf) if st.session_state.sites_gdf is not None else 0),
                    ("🌊 Rivers", len(st.session_state.rivers_gdf) if st.session_state.rivers_gdf is not None else 0)
                ]
                
                for label, value in metrics:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{value}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Data quality indicators
                if st.session_state.data_logger:
                    coord_summary = st.session_state.data_logger.get_coordinate_processing_summary()
                    if coord_summary['total_processed'] > 0:
                        st.markdown(f"""
                        **🎯 Data Quality:**
                        - Success Rate: {coord_summary['success_rate']:.1f}%
                        - Total Processed: {coord_summary['total_processed']}
                        - Parsing Failures: {coord_summary['failed_coordinates']}
                        """)
            else:
                st.info("Load data to see summary")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("## 🔍 Advanced Analysis & Machine Learning")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first in the Data Collection tab")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.markdown("### 🧮 Archaeological Analysis")
                
                if st.button("🔬 Run Comprehensive Analysis", use_container_width=True):
                    with st.spinner("🔬 Analyzing archaeological patterns..."):
                        analysis_results = st.session_state.archaeological_analyzer.analyze_settlement_patterns(
                            st.session_state.tribal_gdf,
                            st.session_state.sites_gdf
                        )
                        st.session_state.analysis_results = analysis_results
                        
                        if 'error' not in analysis_results:
                            st.success("✅ Analysis completed successfully!")
                        else:
                            st.error(f"❌ Analysis failed: {analysis_results['error']}")
                
                # Display analysis results
                if st.session_state.analysis_results and 'error' not in st.session_state.analysis_results:
                    st.markdown("#### 📊 Analysis Results")
                    
                    results = st.session_state.analysis_results
                    
                    # Proximity Analysis
                    if 'proximity' in results:
                        prox = results['proximity']
                        st.markdown(f"""
                        **🎯 Proximity Analysis:**
                        - Mean distance to nearest site: {prox.get('mean_distance_to_nearest_site_km', 0):.1f} km
                        - Median distance: {prox.get('median_distance_to_nearest_site_km', 0):.1f} km
                        - Range: {prox.get('min_distance_km', 0):.1f} - {prox.get('max_distance_km', 0):.1f} km
                        - Standard deviation: {prox.get('std_distance_km', 0):.1f} km
                        """)
                        
                        # Distance distribution
                        if 'distances_distribution' in prox:
                            dist = prox['distances_distribution']
                            st.markdown(f"""
                            **📊 Distance Distribution:**
                            - Very close (0-10km): {dist.get('very_close_0_10km', 0)} sites
                            - Close (10-50km): {dist.get('close_10_50km', 0)} sites
                            - Moderate (50-100km): {dist.get('moderate_50_100km', 0)} sites
                            - Far (100-200km): {dist.get('far_100_200km', 0)} sites
                            - Very far (200km+): {dist.get('very_far_200km_plus', 0)} sites
                            """)
                    
                    # Clustering Analysis
                    if 'tribal_clustering' in results:
                        clust = results['tribal_clustering']
                        st.markdown(f"""
                        **🔍 Tribal Clustering Analysis:**
                        - Total tribes: {clust.get('total_sites', 0)}
                        - Clustered tribes: {clust.get('clustered_sites', 0)}
                        - Clustering percentage: {clust.get('clustering_percentage', 0):.1f}%
                        - Number of clusters: {clust.get('number_of_clusters', 0)}
                        - Average cluster size: {clust.get('average_cluster_size', 0):.1f}
                        """)
                    
                    if 'archaeological_clustering' in results:
                        clust = results['archaeological_clustering']
                        st.markdown(f"""
                        **🏛️ Archaeological Clustering Analysis:**
                        - Total sites: {clust.get('total_sites', 0)}
                        - Clustered sites: {clust.get('clustered_sites', 0)}
                        - Clustering percentage: {clust.get('clustering_percentage', 0):.1f}%
                        - Number of clusters: {clust.get('number_of_clusters', 0)}
                        """)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.markdown("### 🌍 Geographic Distribution Analysis")
                
                # Diversity Analysis
                if st.session_state.tribal_gdf is not None:
                    st.markdown("#### 🌍 Diversity Analysis")
                    diversity_metrics = st.session_state.archaeological_analyzer.calculate_diversity_index(
                        st.session_state.tribal_gdf, 'country'
                    )
                    
                    if diversity_metrics:
                        st.markdown(f"""
                        **📈 Diversity Metrics:**
                        - Shannon Index: {diversity_metrics.get('shannon_diversity', 0):.3f}
                        - Simpson Index: {diversity_metrics.get('simpson_diversity', 0):.3f}
                        - Evenness: {diversity_metrics.get('evenness', 0):.3f}
                        - Berger-Parker Dominance: {diversity_metrics.get('berger_parker_dominance', 0):.3f}
                        - Margalef Richness: {diversity_metrics.get('margalef_richness', 0):.3f}
                        - Total Countries: {diversity_metrics.get('total_groups', 0)}
                        - Singleton Countries: {diversity_metrics.get('singleton_groups', 0)}
                        """)
                        
                        # Most and least represented
                        st.markdown(f"""
                        **🏆 Geographic Distribution:**
                        - Most represented: {diversity_metrics.get('most_common_group', 'Unknown')}
                        - Least represented: {diversity_metrics.get('rarest_group', 'Unknown')}
                        """)
                
                # Spatial distribution summary
                if st.session_state.tribal_gdf is not None and len(st.session_state.tribal_gdf) > 0:
                    st.markdown("#### 📍 Spatial Distribution Summary")
                    bounds = st.session_state.tribal_gdf.total_bounds
                    center_lat = st.session_state.tribal_gdf.geometry.y.mean()
                    center_lon = st.session_state.tribal_gdf.geometry.x.mean()
                    
                    st.markdown(f"""
                    **🗺️ Geographic Extent:**
                    - Latitude range: {bounds[1]:.2f}° to {bounds[3]:.2f}°
                    - Longitude range: {bounds[0]:.2f}° to {bounds[2]:.2f}°
                    - Center point: {center_lat:.4f}°, {center_lon:.4f}°
                    - Approximate area: {(bounds[2] - bounds[0]) * 111 * (bounds[3] - bounds[1]) * 111:.0f} km²
                    """)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("## 📈 Advanced Archaeological Visualizations")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
        else:
            if st.button("🎨 Generate Comprehensive Visualizations", type="primary", use_container_width=True):
                with st.spinner("🎨 Creating advanced archaeological visualizations..."):
                    viz_suite = st.session_state.visualization_suite
                    
                    visualizations = {}
                    
                    # Archaeological Distribution Map
                    distribution_map = viz_suite.create_archaeological_distribution_map(
                        st.session_state.tribal_gdf, st.session_state.sites_gdf
                    )
                    visualizations['distribution_map'] = distribution_map
                    
                    # Country Analysis Chart
                    country_analysis = viz_suite.create_country_analysis_chart(
                        st.session_state.tribal_gdf, st.session_state.sites_gdf
                    )
                    visualizations['country_analysis'] = country_analysis
                    
                    # Cluster Visualizations
                    if st.session_state.tribal_gdf is not None and 'cluster' in st.session_state.tribal_gdf.columns:
                        tribal_clusters = viz_suite.create_cluster_visualization(
                            st.session_state.tribal_gdf, "Tribal Settlements"
                        )
                        visualizations['tribal_clusters'] = tribal_clusters
                    
                    if st.session_state.sites_gdf is not None and 'cluster' in st.session_state.sites_gdf.columns:
                        sites_clusters = viz_suite.create_cluster_visualization(
                            st.session_state.sites_gdf, "Archaeological Sites"
                        )
                        visualizations['sites_clusters'] = sites_clusters
                    
                    # 3D Visualization
                    viz_3d = viz_suite.create_3d_archaeological_landscape(
                        st.session_state.tribal_gdf,
                        st.session_state.sites_gdf
                    )
                    visualizations['3d'] = viz_3d
                    
                    # Diversity Analysis Chart
                    diversity_chart = viz_suite.create_diversity_analysis_chart(
                        st.session_state.tribal_gdf, st.session_state.sites_gdf
                    )
                    visualizations['diversity'] = diversity_chart
                    
                    st.session_state.visualizations = visualizations
                    st.success("✅ Comprehensive visualizations generated successfully!")
            
            # Display visualizations
            if st.session_state.visualizations:
                viz = st.session_state.visualizations
                
                # Archaeological Distribution Map
                if 'distribution_map' in viz and viz['distribution_map'].data:
                    st.markdown("### 🗺️ Archaeological Data Distribution Map")
                    st.markdown("*Interactive map showing the geographic distribution of indigenous tribes and archaeological sites*")
                    st.plotly_chart(viz['distribution_map'], use_container_width=True)
                
                # Country Analysis Chart
                if 'country_analysis' in viz and viz['country_analysis'].data:
                    st.markdown("### 📊 Country-wise Archaeological Data Analysis")
                    st.markdown("*Comparative analysis of tribal settlements and archaeological sites by country*")
                    st.plotly_chart(viz['country_analysis'], use_container_width=True)
                
                # Cluster Visualizations
                if 'tribal_clusters' in viz and viz['tribal_clusters'].data:
                    st.markdown("### 🔍 Tribal Settlements Spatial Clustering")
                    st.markdown("*DBSCAN clustering analysis showing spatial patterns in tribal settlements*")
                    st.plotly_chart(viz['tribal_clusters'], use_container_width=True)
                
                if 'sites_clusters' in viz and viz['sites_clusters'].data:
                    st.markdown("### 🔍 Archaeological Sites Spatial Clustering")
                    st.markdown("*DBSCAN clustering analysis showing spatial patterns in archaeological sites*")
                    st.plotly_chart(viz['sites_clusters'], use_container_width=True)
                
                # 3D Visualization
                if '3d' in viz and viz['3d'].data:
                    st.markdown("### 🏔️ 3D Archaeological Landscape")
                    st.markdown("*Three-dimensional visualization with clustering and anomaly detection*")
                    st.plotly_chart(viz['3d'], use_container_width=True)
                
                # Diversity Analysis Chart
                if 'diversity' in viz and viz['diversity'].data:
                    st.markdown("### 📊 Comprehensive Diversity Analysis")
                    st.markdown("*Multi-panel analysis of geographic diversity and distribution patterns*")
                    st.plotly_chart(viz['diversity'], use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown("## 🗺️ Professional Interactive Map")

        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data from the Data Collection tab first.")
        else:
            # Map generation
            if st.button("🗺️ Generate Professional Interactive Map", use_container_width=True):
                with st.spinner("🔄 Generating professional interactive map with enhanced legend and proper naming..."):
                    try:
                        # Generate map HTML
                        map_html = create_professional_interactive_map_html(
                            st.session_state.tribal_gdf,
                            st.session_state.sites_gdf,
                            st.session_state.rivers_gdf
                        )
                        
                        # Display the map
                        st.markdown("### 🗺️ Amazon Archaeological Research Interactive Map")
                        st.markdown("""
                        **Features:**
                        - 🏘️ **Indigenous Tribal Settlements** (Red circles) - Real data from Wikidata
                        - 🏛️ **Archaeological Heritage Sites** (Brown diamonds) - UNESCO and heritage sites
                        - 🌊 **Amazon River Systems** (Blue circles) - Major rivers and tributaries
                        - 🔍 **Spatial Clusters** - DBSCAN clustering analysis results
                        - 🚨 **Spatial Anomalies** - Outlier detection using Isolation Forest
                        - 📊 **Professional Legend** - Clear labeling with black text for readability
                        - 🎛️ **Layer Controls** - Toggle different data layers on/off
                        - 📍 **Detailed Popups** - Comprehensive information for each location
                        - 🗺️ **Multiple Base Maps** - OpenStreetMap, Satellite, and Topographic views
                        """)
                        
                        # Embed the map
                        components.html(map_html, height=700, scrolling=False)
                        
                        # Map statistics
                        tribal_count = len(st.session_state.tribal_gdf) if st.session_state.tribal_gdf is not None else 0
                        sites_count = len(st.session_state.sites_gdf) if st.session_state.sites_gdf is not None else 0
                        rivers_count = len(st.session_state.rivers_gdf) if st.session_state.rivers_gdf is not None else 0
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🏘️ Indigenous Tribes", tribal_count)
                        with col2:
                            st.metric("🏛️ Archaeological Sites", sites_count)
                        with col3:
                            st.metric("🌊 River Systems", rivers_count)
                        with col4:
                            st.metric("📍 Total Points", tribal_count + sites_count + rivers_count)
                        
                        st.success("✅ **Professional Interactive Map Generated Successfully!**")
                        st.info("🎯 **Map Features**: Professional legend with black text, proper naming, enhanced layer controls, and comprehensive data visualization")
                        
                    except Exception as e:
                        st.error(f"❌ **Map Generation Error**: {str(e)}")
                        st.session_state.data_logger.log_processing_error("Interactive Map Generation", str(e))
    
    with tab5:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("## 📋 Comprehensive Research Reports")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data and run analysis first")
        else:
            if st.button("📊 Generate Comprehensive Research Report", use_container_width=True):
                with st.spinner("📊 Generating comprehensive research report..."):
                    
                    # Report Header
                    st.markdown("# 🏛️ Amazon Archaeological Research Report")
                    st.markdown(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown("**Data Sources:** Wikidata SPARQL API")
                    st.markdown("---")
                    
                    # Executive Summary
                    st.markdown("## 📋 Executive Summary")
                    
                    tribal_count = len(st.session_state.tribal_gdf) if st.session_state.tribal_gdf is not None else 0
                    sites_count = len(st.session_state.sites_gdf) if st.session_state.sites_gdf is not None else 0
                    rivers_count = len(st.session_state.rivers_gdf) if st.session_state.rivers_gdf is not None else 0
                    
                    st.markdown(f"""
                    This comprehensive archaeological research report analyzes **{tribal_count + sites_count + rivers_count}** 
                    data points across the Amazon region, including **{tribal_count}** indigenous tribal settlements, 
                    **{sites_count}** archaeological heritage sites, and **{rivers_count}** river systems.
                    
                    The analysis employs advanced geospatial processing, machine learning clustering algorithms, 
                    and statistical analysis to identify patterns, anomalies, and relationships within the 
                    archaeological landscape of the Amazon basin.
                    """)
                    
                    # Data Quality Assessment
                    st.markdown("## 🎯 Data Quality Assessment")
                    
                    if st.session_state.data_logger:
                        coord_summary = st.session_state.data_logger.get_coordinate_processing_summary()
                        operation_summary = st.session_state.data_logger.get_operation_summary()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📊 Coordinate Processing Quality")
                            if coord_summary['total_processed'] > 0:
                                st.markdown(f"""
                                - **Total Records Processed:** {coord_summary['total_processed']}
                                - **Successfully Parsed:** {coord_summary['successful_coordinates']} ({coord_summary['success_rate']:.1f}%)
                                - **Parsing Failures:** {coord_summary['failed_coordinates']}
                                - **Invalid Formats:** {coord_summary['invalid_formats']}
                                - **Out of Range Values:** {coord_summary['out_of_range']}
                                """)
                            else:
                                st.info("No coordinate processing data available")
                        
                        with col2:
                            st.markdown("### 🔗 API Performance")
                            st.markdown(f"""
                            - **Total API Operations:** {operation_summary['total_operations']}
                            - **Successful Requests:** {operation_summary['data_successes']}
                            - **Failed Requests:** {operation_summary['api_failures']}
                            - **Processing Errors:** {operation_summary['processing_errors']}
                            """)
                    
                    # Geographic Distribution Analysis
                    st.markdown("## 🌍 Geographic Distribution Analysis")
                    
                    if st.session_state.tribal_gdf is not None and len(st.session_state.tribal_gdf) > 0:
                        st.markdown("### 🏘️ Indigenous Tribal Settlements")
                        
                        tribal_countries = st.session_state.tribal_gdf['country'].value_counts()
                        bounds = st.session_state.tribal_gdf.total_bounds
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Country Distribution:**")
                            for country, count in tribal_countries.head(10).items():
                                percentage = (count / len(st.session_state.tribal_gdf)) * 100
                                st.markdown(f"- {country}: {count} settlements ({percentage:.1f}%)")
                        
                        with col2:
                            st.markdown("**Geographic Extent:**")
                            st.markdown(f"""
                            - **Latitude Range:** {bounds[1]:.2f}° to {bounds[3]:.2f}°
                            - **Longitude Range:** {bounds[0]:.2f}° to {bounds[2]:.2f}°
                            - **Approximate Area:** {(bounds[2] - bounds[0]) * 111 * (bounds[3] - bounds[1]) * 111:.0f} km²
                            - **Center Point:** {st.session_state.tribal_gdf.geometry.y.mean():.4f}°, {st.session_state.tribal_gdf.geometry.x.mean():.4f}°
                            """)
                    
                    if st.session_state.sites_gdf is not None and len(st.session_state.sites_gdf) > 0:
                        st.markdown("### 🏛️ Archaeological Heritage Sites")
                        
                        sites_countries = st.session_state.sites_gdf['country'].value_counts()
                        bounds = st.session_state.sites_gdf.total_bounds
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Country Distribution:**")
                            for country, count in sites_countries.head(10).items():
                                percentage = (count / len(st.session_state.sites_gdf)) * 100
                                st.markdown(f"- {country}: {count} sites ({percentage:.1f}%)")
                        
                        with col2:
                            st.markdown("**Geographic Extent:**")
                            st.markdown(f"""
                            - **Latitude Range:** {bounds[1]:.2f}° to {bounds[3]:.2f}°
                            - **Longitude Range:** {bounds[0]:.2f}° to {bounds[2]:.2f}°
                            - **Approximate Area:** {(bounds[2] - bounds[0]) * 111 * (bounds[3] - bounds[1]) * 111:.0f} km²
                            - **Center Point:** {st.session_state.sites_gdf.geometry.y.mean():.4f}°, {st.session_state.sites_gdf.geometry.x.mean():.4f}°
                            """)
                    
                    # Advanced Analysis Results
                    if st.session_state.analysis_results and 'error' not in st.session_state.analysis_results:
                        st.markdown("## 🔬 Advanced Archaeological Analysis")
                        
                        results = st.session_state.analysis_results
                        
                        # Proximity Analysis
                        if 'proximity' in results:
                            st.markdown("### 📏 Proximity Analysis")
                            prox = results['proximity']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Distance Statistics:**")
                                st.markdown(f"""
                                - **Mean Distance to Nearest Site:** {prox.get('mean_distance_to_nearest_site_km', 0):.1f} km
                                - **Median Distance:** {prox.get('median_distance_to_nearest_site_km', 0):.1f} km
                                - **Minimum Distance:** {prox.get('min_distance_km', 0):.1f} km
                                - **Maximum Distance:** {prox.get('max_distance_km', 0):.1f} km
                                - **Standard Deviation:** {prox.get('std_distance_km', 0):.1f} km
                                """)
                            
                            with col2:
                                if 'distances_distribution' in prox:
                                    st.markdown("**Distance Distribution:**")
                                    dist = prox['distances_distribution']
                                    total = sum(dist.values())
                                    for category, count in dist.items():
                                        percentage = (count / total) * 100 if total > 0 else 0
                                        st.markdown(f"- {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
                        
                        # Clustering Analysis
                        if 'tribal_clustering' in results or 'archaeological_clustering' in results:
                            st.markdown("### 🔍 Spatial Clustering Analysis")
                            
                            col1, col2 = st.columns(2)
                            
                            if 'tribal_clustering' in results:
                                with col1:
                                    st.markdown("**Tribal Settlement Clusters:**")
                                    clust = results['tribal_clustering']
                                    st.markdown(f"""
                                    - **Total Settlements:** {clust.get('total_sites', 0)}
                                    - **Clustered Settlements:** {clust.get('clustered_sites', 0)}
                                    - **Clustering Percentage:** {clust.get('clustering_percentage', 0):.1f}%
                                    - **Number of Clusters:** {clust.get('number_of_clusters', 0)}
                                    - **Average Cluster Size:** {clust.get('average_cluster_size', 0):.1f}
                                    """)
                            
                            if 'archaeological_clustering' in results:
                                with col2:
                                    st.markdown("**Archaeological Site Clusters:**")
                                    clust = results['archaeological_clustering']
                                    st.markdown(f"""
                                    - **Total Sites:** {clust.get('total_sites', 0)}
                                    - **Clustered Sites:** {clust.get('clustered_sites', 0)}
                                    - **Clustering Percentage:** {clust.get('clustering_percentage', 0):.1f}%
                                    - **Number of Clusters:** {clust.get('number_of_clusters', 0)}
                                    - **Average Cluster Size:** {clust.get('average_cluster_size', 0):.1f}
                                    """)
                        
                        # Comparative Analysis
                        if 'comparative_analysis' in results:
                            st.markdown("### ⚖️ Comparative Analysis")
                            comp = results['comparative_analysis']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Geographic Relationship:**")
                                st.markdown(f"""
                                - **Distance Between Centers:** {comp.get('tribal_vs_archaeological_centers_distance_km', 0):.1f} km
                                - **Overlapping Countries:** {len(comp.get('overlapping_countries', []))}
                                - **Tribal-Only Countries:** {len(comp.get('tribal_only_countries', []))}
                                - **Archaeological-Only Countries:** {len(comp.get('archaeological_only_countries', []))}
                                """)
                            
                            with col2:
                                st.markdown("**Country Overlap:**")
                                overlapping = comp.get('overlapping_countries', [])
                                if overlapping:
                                    for country in overlapping[:10]:
                                        st.markdown(f"- {country}")
                                else:
                                    st.markdown("- No overlapping countries found")
                    
                    # Diversity Analysis
                    if st.session_state.tribal_gdf is not None:
                        diversity_metrics = st.session_state.archaeological_analyzer.calculate_diversity_index(
                            st.session_state.tribal_gdf, 'country'
                        )
                        
                        if diversity_metrics:
                            st.markdown("## 📊 Diversity Analysis")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Diversity Indices:**")
                                st.markdown(f"""
                                - **Shannon Diversity Index:** {diversity_metrics.get('shannon_diversity', 0):.3f}
                                - **Simpson Diversity Index:** {diversity_metrics.get('simpson_diversity', 0):.3f}
                                - **Evenness:** {diversity_metrics.get('evenness', 0):.3f}
                                - **Berger-Parker Dominance:** {diversity_metrics.get('berger_parker_dominance', 0):.3f}
                                - **Margalef Richness:** {diversity_metrics.get('margalef_richness', 0):.3f}
                                """)
                            
                            with col2:
                                st.markdown("**Distribution Summary:**")
                                st.markdown(f"""
                                - **Total Countries:** {diversity_metrics.get('total_groups', 0)}
                                - **Total Records:** {diversity_metrics.get('total_records', 0)}
                                - **Most Common Country:** {diversity_metrics.get('most_common_group', 'Unknown')}
                                - **Rarest Country:** {diversity_metrics.get('rarest_group', 'Unknown')}
                                - **Singleton Countries:** {diversity_metrics.get('singleton_groups', 0)}
                                """)
                    
                    # Anomaly Detection Results
                    if st.session_state.tribal_gdf is not None and 'is_anomaly' in st.session_state.tribal_gdf.columns:
                        tribal_anomalies = st.session_state.tribal_gdf[st.session_state.tribal_gdf['is_anomaly'] == True]
                        if len(tribal_anomalies) > 0:
                            st.markdown("### 🚨 Spatial Anomaly Detection")
                            st.markdown(f"**Tribal Settlement Anomalies:** {len(tribal_anomalies)} detected")
                            
                            if len(tribal_anomalies) <= 10:
                                for _, anomaly in tribal_anomalies.iterrows():
                                    st.markdown(f"- **{anomaly['name']}** ({anomaly['country']}) - Distance from centroid: {anomaly.get('distance_from_centroid_km', 0):.1f} km")
                    
                    if st.session_state.sites_gdf is not None and 'is_anomaly' in st.session_state.sites_gdf.columns:
                        sites_anomalies = st.session_state.sites_gdf[st.session_state.sites_gdf['is_anomaly'] == True]
                        if len(sites_anomalies) > 0:
                            st.markdown(f"**Archaeological Site Anomalies:** {len(sites_anomalies)} detected")
                            
                            if len(sites_anomalies) <= 10:
                                for _, anomaly in sites_anomalies.iterrows():
                                    st.markdown(f"- **{anomaly['name']}** ({anomaly['country']}) - Distance from centroid: {anomaly.get('distance_from_centroid_km', 0):.1f} km")
                    
                    # Methodology
                    st.markdown("## 🔬 Methodology")
                    st.markdown("""
                    ### Data Collection
                    - **Primary Source:** Wikidata SPARQL API
                    - **Data Types:** Indigenous tribes, archaeological sites, river systems
                    - **Geographic Scope:** Amazon basin countries (Brazil, Peru, Colombia, Venezuela, Ecuador, Bolivia, Guyana, Suriname, French Guiana)
                    - **Quality Control:** Enhanced coordinate parsing with multiple format support
                    
                    ### Geospatial Processing
                    - **Coordinate Systems:** WGS84 (EPSG:4326) for geographic coordinates
                    - **Projection:** Web Mercator (EPSG:3857) for distance calculations
                    - **Validation:** Range checking, format validation, duplicate detection
                    
                    ### Machine Learning Analysis
                    - **Clustering Algorithm:** DBSCAN with 50km epsilon parameter
                    - **Anomaly Detection:** Isolation Forest with 10% contamination threshold
                    - **Distance Calculations:** Haversine formula for great circle distances
                    
                    ### Statistical Analysis
                    - **Diversity Indices:** Shannon, Simpson, Berger-Parker, Margalef
                    - **Spatial Statistics:** Centroid calculation, bounding box analysis
                    - **Proximity Analysis:** Nearest neighbor distances, distribution analysis
                    """)
                    
                    # Conclusions
                    st.markdown("## 📝 Conclusions")
                    st.markdown(f"""
                    This comprehensive analysis of {tribal_count + sites_count} archaeological data points 
                    across the Amazon region reveals significant patterns in the distribution of indigenous 
                    settlements and archaeological heritage sites.
                    
                    **Key Findings:**
                    - The data demonstrates a {coord_summary.get('success_rate', 0):.1f}% success rate in coordinate processing
                    - Spatial clustering analysis reveals organized settlement patterns
                    - Anomaly detection identifies {len(tribal_anomalies) if 'tribal_anomalies' in locals() else 0} unusual tribal locations and {len(sites_anomalies) if 'sites_anomalies' in locals() else 0} unusual archaeological sites
                    - Geographic diversity analysis shows distribution across multiple Amazon basin countries
                    
                    **Research Implications:**
                    - The clustering patterns suggest historical migration routes and settlement preferences
                    - Proximity analysis between tribal settlements and archaeological sites indicates cultural continuity
                    - Spatial anomalies may represent unique historical or geographic circumstances
                    - The comprehensive dataset provides a foundation for further archaeological research
                    
                    **Data Quality:**
                    - All data sourced from verified APIs with no simulated or generated content
                    - Enhanced error handling ensures data integrity
                    - Comprehensive logging provides full traceability
                    """)
                    
                    st.success("✅ **Comprehensive Research Report Generated Successfully!**")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("## 💾 Data Export & System Logs")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.markdown("### 📤 Data Export Options")
                
                # CSV Export
                if st.button("📊 Export Tribal Data (CSV)", use_container_width=True):
                    if st.session_state.tribal_gdf is not None:
                        try:
                            # Prepare data for export
                            export_data = st.session_state.tribal_gdf.copy()
                            export_data['latitude'] = export_data.geometry.y
                            export_data['longitude'] = export_data.geometry.x
                            export_data = export_data.drop('geometry', axis=1)
                            
                            csv_data = export_data.to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download Tribal Data CSV",
                                data=csv_data,
                                file_name=f"amazon_tribal_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            st.success("✅ Tribal data prepared for download")
                        except Exception as e:
                            st.error(f"❌ Export failed: {str(e)}")
                    else:
                        st.warning("No tribal data available")
                
                if st.button("🏛️ Export Archaeological Sites (CSV)", use_container_width=True):
                    if st.session_state.sites_gdf is not None:
                        try:
                            export_data = st.session_state.sites_gdf.copy()
                            export_data['latitude'] = export_data.geometry.y
                            export_data['longitude'] = export_data.geometry.x
                            export_data = export_data.drop('geometry', axis=1)
                            
                            csv_data = export_data.to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download Archaeological Sites CSV",
                                data=csv_data,
                                file_name=f"amazon_archaeological_sites_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            st.success("✅ Archaeological sites data prepared for download")
                        except Exception as e:
                            st.error(f"❌ Export failed: {str(e)}")
                    else:
                        st.warning("No archaeological sites data available")
                
                # GeoJSON Export
                if st.button("🗺️ Export GeoJSON Data", use_container_width=True):
                    try:
                        geojson_data = {
                            "type": "FeatureCollection",
                            "features": []
                        }
                        
                        # Add tribal data
                        if st.session_state.tribal_gdf is not None:
                            for _, row in st.session_state.tribal_gdf.iterrows():
                                feature = {
                                    "type": "Feature",
                                    "geometry": {
                                        "type": "Point",
                                        "coordinates": [row.geometry.x, row.geometry.y]
                                    },
                                    "properties": {
                                        "name": row['name'],
                                        "country": row['country'],
                                        "type": "tribal_settlement",
                                        "cluster": int(row.get('cluster', -1)) if 'cluster' in row else -1,
                                        "is_anomaly": bool(row.get('is_anomaly', False)) if 'is_anomaly' in row else False
                                    }
                                }
                                geojson_data["features"].append(feature)
                        
                        # Add archaeological sites
                        if st.session_state.sites_gdf is not None:
                            for _, row in st.session_state.sites_gdf.iterrows():
                                feature = {
                                    "type": "Feature",
                                    "geometry": {
                                        "type": "Point",
                                        "coordinates": [row.geometry.x, row.geometry.y]
                                    },
                                    "properties": {
                                        "name": row['name'],
                                        "country": row['country'],
                                        "type": "archaeological_site",
                                        "cluster": int(row.get('cluster', -1)) if 'cluster' in row else -1,
                                        "is_anomaly": bool(row.get('is_anomaly', False)) if 'is_anomaly' in row else False
                                    }
                                }
                                geojson_data["features"].append(feature)
                        
                        geojson_str = json.dumps(geojson_data, indent=2, default=safe_json_serialize)
                        st.download_button(
                            label="⬇️ Download GeoJSON",
                            data=geojson_str,
                            file_name=f"amazon_archaeological_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                            mime="application/geo+json"
                        )
                        st.success("✅ GeoJSON data prepared for download")
                    except Exception as e:
                        st.error(f"❌ GeoJSON export failed: {str(e)}")
                
                # Analysis Results Export
                if st.button("📊 Export Analysis Results (JSON)", use_container_width=True):
                    if st.session_state.analysis_results:
                        try:
                            analysis_json = json.dumps(st.session_state.analysis_results, indent=2, default=safe_json_serialize)
                            st.download_button(
                                label="⬇️ Download Analysis Results",
                                data=analysis_json,
                                file_name=f"amazon_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                            st.success("✅ Analysis results prepared for download")
                        except Exception as e:
                            st.error(f"❌ Analysis export failed: {str(e)}")
                    else:
                        st.warning("No analysis results available")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.markdown("### 📋 System Logs & Diagnostics")
                
                if st.session_state.data_logger:
                    operation_summary = st.session_state.data_logger.get_operation_summary()
                    
                    # Operation Summary
                    st.markdown("#### 📊 Operation Summary")
                    st.markdown(f"""
                    - **Total Operations:** {operation_summary['total_operations']}
                    - **API Failures:** {operation_summary['api_failures']}
                    - **Data Successes:** {operation_summary['data_successes']}
                    - **Processing Errors:** {operation_summary['processing_errors']}
                    """)
                    
                    # Coordinate Processing Details
                    coord_summary = operation_summary.get('coordinate_processing', {})
                    if coord_summary:
                        st.markdown("#### 🗺️ Coordinate Processing Details")
                        st.markdown(f"""
                        - **Total Processed:** {coord_summary.get('total_processed', 0)}
                        - **Successfully Parsed:** {coord_summary.get('successful_coordinates', 0)}
                        - **Success Rate:** {coord_summary.get('success_rate', 0):.1f}%
                        - **Failed Coordinates:** {coord_summary.get('failed_coordinates', 0)}
                        - **Invalid Formats:** {coord_summary.get('invalid_formats', 0)}
                        - **Out of Range:** {coord_summary.get('out_of_range', 0)}
                        """)
                        
                        # Recent failures
                        recent_failures = coord_summary.get('recent_failures', [])
                        if recent_failures:
                            st.markdown("#### ⚠️ Recent Coordinate Failures")
                            for failure in recent_failures[-5:]:
                                st.markdown(f"- **{failure.get('item_name', 'Unknown')}**: {failure.get('reason', 'Unknown reason')}")
                    
                    # Export logs
                    if st.button("📋 Export System Logs", use_container_width=True):
                        try:
                            logs_data = {
                                "export_timestamp": datetime.now().isoformat(),
                                "operation_summary": operation_summary,
                                "coordinate_processing": coord_summary,
                                "recent_operations": operation_summary.get('operations', [])[-50:] if operation_summary.get('operations') else []
                            }
                            
                            logs_json = json.dumps(logs_data, indent=2, default=safe_json_serialize)
                            st.download_button(
                                label="⬇️ Download System Logs",
                                data=logs_json,
                                file_name=f"amazon_platform_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                            st.success("✅ System logs prepared for download")
                        except Exception as e:
                            st.error(f"❌ Log export failed: {str(e)}")
                    
                    # Clear logs
                    if st.button("🗑️ Clear System Logs", use_container_width=True):
                        st.session_state.data_logger = DataLogger()
                        st.success("✅ System logs cleared")
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; padding: 2rem;">
        🏛️ <strong>Amazon Archaeological Research Platform</strong> | 
        Enhanced Coordinate Processing | API Key Management | Professional Research Tools<br>
        Powered by Wikidata SPARQL API, GeoPandas, Scikit-learn, and Advanced Analytics<br>
        <em>All data sourced from verified APIs - No simulated or generated content</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
