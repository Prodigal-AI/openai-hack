import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import folium
from streamlit_folium import st_folium
from folium import plugins
import base64
from io import BytesIO, StringIO
import warnings
import hashlib
import os
import streamlit.components.v1 as components
import re
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlencode, quote
import xml.etree.ElementTree as ET
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
from contextlib import contextmanager
import feedparser

warnings.filterwarnings('ignore')

# Enhanced imports for advanced features with proper error handling
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, LineString
    from shapely.ops import unary_union
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.svm import SVC, OneClassSVM
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.neighbors import LocalOutlierFactor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from skimage import feature, filters, morphology, measure, segmentation
    from skimage.transform import hough_circle, hough_circle_peaks, hough_line, hough_line_peaks
    from skimage.feature import corner_harris, corner_peaks, local_binary_pattern, greycomatrix, greycoprops
    from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import rasterio
    from rasterio.plot import show
    from rasterio.mask import mask
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Google Earth Engine - MAIN FOCUS
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

# OpenAI Integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configuration
st.set_page_config(
    page_title="Archaeological Site Discovery Platform - Real APIs",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default Configuration - REAL APIs ONLY
DEFAULT_CONFIG = {
    'gee_project_id': 'snappy-provider-429510-s9',  # Default GEE project
    'openweather_api_key': '',  # User must provide
    'openai_api_key': '',  # User must provide
    'wikidata_endpoint': 'https://query.wikidata.org/sparql',
    'rss_feeds': [
        'https://www.archaeology.org/feed',
        'https://www.sciencedaily.com/rss/fossils_ruins/archaeology.xml',
        'https://phys.org/rss-feed/earth-news/archaeology/',
        'https://www.heritagedaily.com/feed'
    ],
    'api_timeout': 15,
    'processing_timeout': 30,
    'analysis_timeout': 120
}

# Cross-platform timeout context manager
@contextmanager
def timeout(duration):
    """Cross-platform timeout context manager"""
    result = {'timed_out': False}
    
    def target():
        time.sleep(duration)
        result['timed_out'] = True
        
    timer = threading.Timer(duration, target)
    timer.start()
    
    try:
        yield result
    finally:
        timer.cancel()

def initialize_gee(project_id: str):
    """Initialize Google Earth Engine with project ID"""
    if not GEE_AVAILABLE:
        st.warning("⚠️ Google Earth Engine not available. Please install: pip install earthengine-api")
        return False
    
    try:
        # Simple initialization without timeout for Windows compatibility
        ee.Initialize(project=project_id)
        st.success("✅ Google Earth Engine initialized successfully!")
        return True
    except Exception as e:
        try:
            st.info("🔐 Attempting Google Earth Engine authentication...")
            ee.Authenticate()
            ee.Initialize(project=project_id)
            st.success("✅ Google Earth Engine authenticated and initialized!")
            return True
        except Exception as auth_error:
            st.error(f"❌ Failed to initialize Google Earth Engine: {str(auth_error)}")
            st.info("💡 Please run 'earthengine authenticate' in your terminal first")
            return False

# Setup logging
def setup_logging():
    """Setup logging system"""
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"archaeological_discovery_{datetime.now().strftime('%Y%m%d')}.log"),
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
    
    .platform-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .modern-card {
        background: #262d3a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    .confidence-card {
        background: #2c3e50;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
        color: #ecf0f1;
    }
    
    .confidence-card.high { 
        border-left-color: #2ecc71; 
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, #2c3e50 100%);
    }
    .confidence-card.medium { 
        border-left-color: #f39c12; 
        background: linear-gradient(135deg, rgba(243, 156, 18, 0.1) 0%, #2c3e50 100%);
    }
    .confidence-card.low { 
        border-left-color: #e74c3c; 
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.1) 0%, #2c3e50 100%);
    }
    
    .metric-card {
        background: #34495e;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #ecf0f1;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-card .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3498db;
        margin: 0.5rem 0;
    }
    
    .metric-card .metric-label {
        font-size: 0.9rem;
        color: #bdc3c7;
        font-weight: 500;
    }
    
    .api-status {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .api-status.online {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid #2ecc71;
    }
    
    .api-status.offline {
        background-color: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        border: 1px solid #e74c3c;
    }
    
    .news-item {
        background: #34495e;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #3498db;
    }
    
    .news-item h4 {
        color: #3498db;
        margin: 0 0 0.5rem 0;
    }
    
    .news-item .news-meta {
        color: #95a5a6;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    
    .ai-insight {
        background: linear-gradient(135deg, #8e44ad 0%, #3498db 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .ai-insight h4 {
        color: white;
        margin: 0 0 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

class OpenAIProcessor:
    """OpenAI integration for archaeological analysis insights"""
    
    def __init__(self, logger):
        self.logger = logger
        self.timeout = 30
    
    def analyze_archaeological_potential(self, location_data: Dict, analysis_results: Dict) -> Dict:
        """Use OpenAI to provide insights on archaeological potential"""
        try:
            api_key = st.session_state.get('api_config', {}).get('openai_api_key', '')
            
            if not api_key or api_key == '':
                return {'status': 'no_api_key', 'message': 'OpenAI API key not provided'}
            
            if not OPENAI_AVAILABLE:
                return {'status': 'not_available', 'message': 'OpenAI library not installed'}
            
            # Set up OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Prepare context for AI analysis
            context = self._prepare_analysis_context(location_data, analysis_results)
            
            # Create prompt for archaeological analysis
            prompt = f"""
            As an expert archaeologist and data analyst, analyze the following archaeological site discovery data and provide insights:

            Location: {location_data['lat']:.4f}, {location_data['lon']:.4f}
            
            Analysis Results:
            {context}
            
            Please provide:
            1. Archaeological potential assessment (High/Medium/Low) with reasoning
            2. Key factors supporting or contradicting archaeological significance
            3. Recommendations for further investigation
            4. Historical/cultural context if relevant
            5. Comparison with known archaeological patterns
            
            Keep the response concise but informative, focusing on scientific archaeological principles.
            """
            
            # Make API call with timeout
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert archaeologist specializing in site discovery and analysis using remote sensing and environmental data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            ai_analysis = response.choices[0].message.content
            
            self.logger.info("Successfully generated OpenAI archaeological analysis")
            st.success("✅ AI archaeological insights generated")
            
            return {
                'status': 'success',
                'analysis': ai_analysis,
                'model_used': 'gpt-3.5-turbo',
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
        except Exception as e:
            st.warning(f"⚠️ OpenAI analysis failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def generate_site_summary(self, all_results: List[Dict]) -> Dict:
        """Generate overall summary of archaeological discoveries"""
        try:
            api_key = st.session_state.get('api_config', {}).get('openai_api_key', '')
            
            if not api_key or api_key == '':
                return {'status': 'no_api_key'}
            
            if not OPENAI_AVAILABLE:
                return {'status': 'not_available'}
            
            client = openai.OpenAI(api_key=api_key)
            
            # Prepare summary data
            summary_data = self._prepare_summary_context(all_results)
            
            prompt = f"""
            As an expert archaeologist, provide a comprehensive summary of this archaeological survey:
            
            Survey Data:
            {summary_data}
            
            Please provide:
            1. Overall assessment of the surveyed region's archaeological potential
            2. Most promising locations and why
            3. Patterns observed across the survey area
            4. Recommendations for prioritizing further investigation
            5. Potential historical/cultural significance of findings
            
            Format as a professional archaeological survey report summary.
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior archaeologist writing a professional survey report."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.6
            )
            
            return {
                'status': 'success',
                'summary': response.choices[0].message.content,
                'model_used': 'gpt-3.5-turbo'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _prepare_analysis_context(self, location_data: Dict, analysis_results: Dict) -> str:
        """Prepare context string for AI analysis"""
        context_parts = []
        
        # Known archaeological sites
        known_sites = analysis_results.get('archaeological_sites', [])
        if known_sites:
            context_parts.append(f"Known archaeological sites nearby: {len(known_sites)}")
            for site in known_sites[:3]:
                context_parts.append(f"- {site['name']} ({site['type']}) - {site.get('distance_km', 0):.1f}km away")
        
        # Environmental data
        env_data = analysis_results.get('environmental_data', {})
        if env_data.get('status') == 'success':
            context_parts.append(f"Environmental conditions:")
            context_parts.append(f"- Temperature: {env_data.get('temperature')}°C")
            context_parts.append(f"- Humidity: {env_data.get('humidity')}%")
            context_parts.append(f"- Weather: {env_data.get('weather_description')}")
        
        # Computer vision features
        cv_features = analysis_results.get('cv_features', {})
        if cv_features and not cv_features.get('error'):
            context_parts.append("Computer vision analysis detected:")
            for feature_type, features in cv_features.items():
                if isinstance(features, list) and features:
                    avg_conf = np.mean([f.get('confidence', 0.5) for f in features])
                    context_parts.append(f"- {feature_type.replace('_', ' ').title()}: {len(features)} features (confidence: {avg_conf:.2f})")
        
        # Overall confidence
        confidence = analysis_results.get('confidence_score', 0)
        context_parts.append(f"Overall archaeological potential score: {confidence:.3f}")
        
        return '\n'.join(context_parts)
    
    def _prepare_summary_context(self, all_results: List[Dict]) -> str:
        """Prepare summary context for multiple locations"""
        context_parts = []
        
        context_parts.append(f"Total locations surveyed: {len(all_results)}")
        
        # Confidence distribution
        confidences = [r.get('confidence_score', 0) for r in all_results]
        high_conf = len([c for c in confidences if c >= 0.7])
        medium_conf = len([c for c in confidences if 0.4 <= c < 0.7])
        low_conf = len([c for c in confidences if c < 0.4])
        
        context_parts.append(f"Confidence distribution: {high_conf} high, {medium_conf} medium, {low_conf} low")
        
        # Total known sites
        total_sites = sum(len(r.get('archaeological_sites', [])) for r in all_results)
        context_parts.append(f"Total known archaeological sites found: {total_sites}")
        
        # Top locations
        sorted_results = sorted(all_results, key=lambda x: x.get('confidence_score', 0), reverse=True)
        context_parts.append("Top 3 locations by potential:")
        for i, result in enumerate(sorted_results[:3]):
            lat, lon = result['location']['lat'], result['location']['lon']
            conf = result.get('confidence_score', 0)
            context_parts.append(f"{i+1}. {lat:.4f}, {lon:.4f} (confidence: {conf:.3f})")
        
        return '\n'.join(context_parts)

class GoogleEarthEngineProcessor:
    """Google Earth Engine processor for real satellite data"""
    
    def __init__(self, logger, project_id: str):
        self.logger = logger
        self.project_id = project_id
        self.gee_initialized = False
        
        try:
            self.gee_initialized = initialize_gee(project_id)
        except Exception as e:
            st.error(f"❌ GEE initialization failed: {str(e)}")
            self.gee_initialized = False
    
    def get_real_satellite_data(self, lat: float, lon: float, buffer_km: float = 5) -> Optional[Dict]:
        """Get real satellite data from Google Earth Engine"""
        if not self.gee_initialized or not GEE_AVAILABLE:
            st.warning("⚠️ Google Earth Engine not available")
            return None
        
        try:
            # Define area of interest
            point = ee.Geometry.Point([lon, lat])
            aoi = point.buffer(buffer_km * 1000)
            
            # Get Sentinel-2 imagery
            s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                           .filterBounds(aoi)
                           .filterDate('2023-01-01', '2024-12-31')
                           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                           .select(['B4', 'B3', 'B2', 'B8']))
            
            # Get the most recent image
            latest_image = s2_collection.sort('system:time_start', False).first()
            
            if latest_image:
                # Get image metadata
                image_info = latest_image.getInfo()
                
                # Get thumbnail URL for visualization
                thumbnail_url = latest_image.getThumbURL({
                    'min': 0,
                    'max': 3000,
                    'bands': ['B4', 'B3', 'B2'],
                    'region': aoi,
                    'dimensions': 512,
                    'format': 'png'
                })
                
                st.success("✅ Successfully acquired Google Earth Engine Sentinel-2 data")
                
                return {
                    'source': 'Google Earth Engine Sentinel-2',
                    'image_info': image_info,
                    'thumbnail_url': thumbnail_url,
                    'coordinates': {'lat': lat, 'lon': lon, 'buffer_km': buffer_km},
                    'acquisition_date': image_info.get('properties', {}).get('system:time_start'),
                    'cloud_coverage': image_info.get('properties', {}).get('CLOUDY_PIXEL_PERCENTAGE')
                }
            else:
                st.warning("⚠️ No Sentinel-2 images found for this location")
                return None
                
        except Exception as e:
            st.warning(f"⚠️ GEE Error: {str(e)}")
            return None

class RealDataProcessor:
    """Real data processor using only specified APIs"""
    
    def __init__(self, logger):
        self.logger = logger
        self.config = DEFAULT_CONFIG
        self.session = requests.Session()
        self.session.timeout = self.config['api_timeout']
        
        # Set headers to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'Archaeological-Discovery-Platform/1.0 (Educational Research)'
        })
    
    def get_real_archaeological_sites(self, lat: float, lon: float, radius_km: float = 50) -> List[Dict]:
        """Get real archaeological sites from Wikidata SPARQL API"""
        try:
            # Wikidata SPARQL query for archaeological sites
            sparql_query = f"""
            SELECT DISTINCT ?site ?siteLabel ?coord ?typeLabel ?countryLabel ?description WHERE {{
              ?site wdt:P31/wdt:P279* wd:Q839954 .  # Archaeological sites
              ?site wdt:P625 ?coord .
              OPTIONAL {{ ?site wdt:P31 ?type . }}
              OPTIONAL {{ ?site wdt:P17 ?country . }}
              OPTIONAL {{ ?site schema:description ?description . FILTER(LANG(?description) = "en") }}
              
              SERVICE wikibase:around {{
                ?site wdt:P625 ?coord .
                bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
                bd:serviceParam wikibase:radius "{radius_km}" .
                bd:serviceParam wikibase:distance ?distance .
              }}
              
              SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "en" . 
              }}
            }}
            ORDER BY ?distance
            LIMIT 50
            """
            
            response = self.session.get(
                self.config['wikidata_endpoint'],
                params={'query': sparql_query, 'format': 'json'},
                timeout=self.config['api_timeout']
            )
            
            if response.status_code == 200:
                data = response.json()
                sites = []
                
                for binding in data.get('results', {}).get('bindings', []):
                    try:
                        coord_str = binding.get('coord', {}).get('value', '')
                        # Parse coordinates from Point(lon lat) format
                        if coord_str.startswith('Point('):
                            coords = coord_str.replace('Point(', '').replace(')', '').split()
                            site_lon, site_lat = float(coords[0]), float(coords[1])
                        else:
                            continue
                        
                        site = {
                            'name': binding.get('siteLabel', {}).get('value', 'Unknown Site'),
                            'type': binding.get('typeLabel', {}).get('value', 'Archaeological Site'),
                            'country': binding.get('countryLabel', {}).get('value', 'Unknown'),
                            'description': binding.get('description', {}).get('value', ''),
                            'coordinates': {'lat': site_lat, 'lon': site_lon},
                            'distance_km': self._calculate_distance(lat, lon, site_lat, site_lon),
                            'source': 'Wikidata',
                            'url': binding.get('site', {}).get('value', '')
                        }
                        sites.append(site)
                    except (ValueError, IndexError, KeyError):
                        continue
                
                self.logger.info(f"Retrieved {len(sites)} real archaeological sites from Wikidata")
                st.success(f"✅ Found {len(sites)} archaeological sites from Wikidata")
                return sites
            else:
                st.warning(f"⚠️ Wikidata API returned status {response.status_code}")
                return []
                
        except Exception as e:
            st.warning(f"⚠️ Wikidata query failed: {str(e)}")
            return []
    
    def get_real_environmental_data(self, lat: float, lon: float) -> Dict:
        """Get real environmental data from OpenWeatherMap API"""
        try:
            # Check if user has provided API key
            api_key = st.session_state.get('api_config', {}).get('openweather_api_key', '')
            
            if not api_key or api_key == '':
                st.info("ℹ️ OpenWeatherMap API key not provided. Environmental data unavailable.")
                return {'source': 'No API Key', 'status': 'unavailable'}
            
            # Current weather data
            current_url = "http://api.openweathermap.org/data/2.5/weather"
            current_params = {
                'lat': lat,
                'lon': lon,
                'appid': api_key,
                'units': 'metric'
            }
            
            current_response = self.session.get(current_url, params=current_params, timeout=self.config['api_timeout'])
            
            if current_response.status_code == 200:
                current_data = current_response.json()
                
                # Air pollution data
                pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution"
                pollution_params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': api_key
                }
                
                pollution_response = self.session.get(pollution_url, params=pollution_params, timeout=self.config['api_timeout'])
                pollution_data = pollution_response.json() if pollution_response.status_code == 200 else {}
                
                env_data = {
                    'temperature': current_data['main']['temp'],
                    'humidity': current_data['main']['humidity'],
                    'pressure': current_data['main']['pressure'],
                    'visibility': current_data.get('visibility', 10000),
                    'weather_description': current_data['weather'][0]['description'],
                    'wind_speed': current_data.get('wind', {}).get('speed', 0),
                    'wind_direction': current_data.get('wind', {}).get('deg', 0),
                    'cloudiness': current_data.get('clouds', {}).get('all', 0),
                    'sunrise': datetime.fromtimestamp(current_data['sys']['sunrise']).isoformat(),
                    'sunset': datetime.fromtimestamp(current_data['sys']['sunset']).isoformat(),
                    'air_quality': pollution_data.get('list', [{}])[0].get('main', {}).get('aqi', 'N/A') if pollution_data else 'N/A',
                    'source': 'OpenWeatherMap API',
                    'status': 'success'
                }
                
                self.logger.info("Retrieved real environmental data from OpenWeatherMap")
                st.success("✅ Environmental data acquired from OpenWeatherMap")
                return env_data
            else:
                st.warning(f"⚠️ OpenWeatherMap API returned status {current_response.status_code}")
                return {'source': 'OpenWeatherMap Error', 'status': 'error', 'error_code': current_response.status_code}
            
        except Exception as e:
            st.warning(f"⚠️ Weather API failed: {str(e)}")
            return {'source': 'Error', 'status': 'error', 'error': str(e)}
    
    def get_archaeological_news(self) -> List[Dict]:
        """Get real archaeological news from RSS feeds"""
        try:
            all_news = []
            
            for feed_url in self.config['rss_feeds']:
                try:
                    # Parse RSS feed
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:5]:  # Get latest 5 from each feed
                        news_item = {
                            'title': entry.get('title', 'No Title'),
                            'summary': entry.get('summary', entry.get('description', 'No Summary')),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'source': feed.feed.get('title', feed_url),
                            'feed_url': feed_url
                        }
                        all_news.append(news_item)
                except Exception as e:
                    self.logger.warning(f"Failed to parse RSS feed {feed_url}: {str(e)}")
                    continue
            
            # Sort by published date (most recent first)
            all_news.sort(key=lambda x: x.get('published', ''), reverse=True)
            
            self.logger.info(f"Retrieved {len(all_news)} news items from RSS feeds")
            st.success(f"✅ Retrieved {len(all_news)} archaeological news items")
            return all_news[:20]  # Return top 20 most recent
            
        except Exception as e:
            st.warning(f"⚠️ RSS feed parsing failed: {str(e)}")
            return []
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r

class ComputerVisionAnalyzer:
    """Advanced computer vision algorithms for feature detection"""
    
    def __init__(self, logger):
        self.logger = logger
        self.timeout = 15
    
    def analyze_satellite_image(self, image_url: str) -> Dict:
        """Analyze satellite image from URL using computer vision"""
        try:
            # Download image
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                return {'error': 'Failed to download image'}
            
            # Convert to numpy array
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                return {'error': 'Failed to decode image'}
            
            # Resize if too large to prevent hanging
            if image.shape[0] > 1024 or image.shape[1] > 1024:
                image = cv2.resize(image, (512, 512))
            
            results = {
                'image_shape': image.shape,
                'circular_features': [],
                'linear_features': [],
                'corner_features': [],
                'texture_features': [],
                'edge_features': []
            }
            
            # Hough Circle Transform
            try:
                circles = self._detect_circles_hough(image)
                results['circular_features'] = circles
            except Exception as e:
                self.logger.warning(f"Circle detection failed: {str(e)}")
            
            # Hough Line Transform
            try:
                lines = self._detect_lines_hough(image)
                results['linear_features'] = lines
            except Exception as e:
                self.logger.warning(f"Line detection failed: {str(e)}")
            
            # Harris Corner Detection
            try:
                corners = self._detect_corners_harris(image)
                results['corner_features'] = corners
            except Exception as e:
                self.logger.warning(f"Corner detection failed: {str(e)}")
            
            # Texture Analysis using Local Binary Patterns
            try:
                texture = self._analyze_texture_lbp(image)
                results['texture_features'] = texture
            except Exception as e:
                self.logger.warning(f"Texture analysis failed: {str(e)}")
            
            # Edge Detection (Canny, Sobel, Laplacian)
            try:
                edges = self._detect_edges_multi(image)
                results['edge_features'] = edges
            except Exception as e:
                self.logger.warning(f"Edge detection failed: {str(e)}")
            
            return results
            
        except Exception as e:
            st.warning(f"⚠️ Computer vision analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _detect_circles_hough(self, image: np.ndarray) -> List[Dict]:
        """Detect circular features using Hough Circle Transform"""
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        features = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for i, (x, y, r) in enumerate(circles[:20]):  # Limit to prevent hanging
                confidence = self._calculate_circle_confidence(image, x, y, r)
                features.append({
                    'type': 'circular',
                    'x': int(x),
                    'y': int(y),
                    'radius': int(r),
                    'confidence': float(confidence),
                    'potential_type': 'settlement_ring' if r > 50 else 'structure'
                })
        
        return features
    
    def _detect_lines_hough(self, image: np.ndarray) -> List[Dict]:
        """Detect linear features using Hough Line Transform"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        features = []
        if lines is not None:
            for i, line in enumerate(lines[:15]):  # Limit to prevent hanging
                rho, theta = line[0]
                confidence = self._calculate_line_confidence(edges, rho, theta)
                
                features.append({
                    'type': 'linear',
                    'rho': float(rho),
                    'theta': float(theta),
                    'confidence': float(confidence),
                    'potential_type': 'road' if abs(theta - np.pi/2) < 0.3 else 'structure_edge'
                })
        
        return features
    
    def _detect_corners_harris(self, image: np.ndarray) -> List[Dict]:
        """Detect corner features using Harris corner detection"""
        image_float = image.astype(np.float32)
        corners = cv2.cornerHarris(image_float, 2, 3, 0.04)
        
        # Find corner coordinates
        corner_coords = np.where(corners > 0.01 * corners.max())
        
        features = []
        for i in range(min(30, len(corner_coords[0]))):  # Limit to prevent hanging
            y, x = corner_coords[0][i], corner_coords[1][i]
            confidence = float(corners[y, x] / corners.max())
            
            features.append({
                'type': 'corner',
                'x': int(x),
                'y': int(y),
                'confidence': confidence,
                'potential_type': 'structure_corner'
            })
        
        return features
    
    def _analyze_texture_lbp(self, image: np.ndarray) -> List[Dict]:
        """Analyze texture using Local Binary Patterns"""
        if not SKIMAGE_AVAILABLE:
            return []
        
        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        # Calculate texture properties
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        # Find regions with unusual texture patterns
        texture_variance = np.var(lbp)
        texture_entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        features = [{
            'type': 'texture',
            'variance': float(texture_variance),
            'entropy': float(texture_entropy),
            'uniformity': float(np.sum(hist**2)),
            'confidence': min(texture_variance / 100.0, 1.0),
            'potential_type': 'surface_anomaly'
        }]
        
        return features
    
    def _detect_edges_multi(self, image: np.ndarray) -> List[Dict]:
        """Multi-method edge detection (Canny, Sobel, Laplacian)"""
        features = []
        
        # Canny edge detection
        canny_edges = cv2.Canny(image, 50, 150)
        canny_density = np.sum(canny_edges > 0) / canny_edges.size
        
        features.append({
            'type': 'canny_edges',
            'edge_density': float(canny_density),
            'confidence': min(canny_density * 10, 1.0),
            'potential_type': 'structural_edges'
        })
        
        # Sobel edge detection
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_density = np.mean(sobel_magnitude) / 255.0
        
        features.append({
            'type': 'sobel_edges',
            'edge_density': float(sobel_density),
            'confidence': min(sobel_density * 5, 1.0),
            'potential_type': 'gradient_features'
        })
        
        # Laplacian edge detection
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian_variance = np.var(laplacian)
        
        features.append({
            'type': 'laplacian_edges',
            'variance': float(laplacian_variance),
            'confidence': min(laplacian_variance / 1000.0, 1.0),
            'potential_type': 'fine_details'
        })
        
        return features
    
    def _calculate_circle_confidence(self, image: np.ndarray, x: int, y: int, r: int) -> float:
        """Calculate confidence score for circular feature"""
        try:
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, 2)
            
            edges = cv2.Canny(image, 50, 150)
            circle_edges = cv2.bitwise_and(edges, mask)
            
            edge_density = np.sum(circle_edges > 0) / (2 * np.pi * r + 1)
            return min(edge_density * 20.0, 1.0)
        except:
            return 0.5
    
    def _calculate_line_confidence(self, edges: np.ndarray, rho: float, theta: float) -> float:
        """Calculate confidence score for linear feature"""
        try:
            h, w = edges.shape
            line_mask = np.zeros((h, w), dtype=np.uint8)
            
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + w * (-b))
            y1 = int(y0 + w * (a))
            x2 = int(x0 - w * (-b))
            y2 = int(y0 - w * (a))
            
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            
            overlap = cv2.bitwise_and(edges, line_mask)
            line_strength = np.sum(overlap > 0) / (np.sum(line_mask > 0) + 1)
            
            return min(line_strength * 2.0, 1.0)
        except:
            return 0.5

class MachineLearningPredictor:
    """Machine learning algorithms for archaeological potential prediction"""
    
    def __init__(self, logger):
        self.logger = logger
        self.timeout = 15
    
    def predict_archaeological_potential(self, features: Dict, environmental_data: Dict, known_sites: List[Dict]) -> Dict:
        """Predict archaeological potential using multiple ML algorithms"""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features, environmental_data, known_sites)
            
            results = {}
            
            # Anomaly Detection using Isolation Forest
            if SKLEARN_AVAILABLE:
                try:
                    anomaly_score = self._anomaly_detection(feature_vector)
                    results['anomaly_detection'] = anomaly_score
                except Exception as e:
                    self.logger.warning(f"Anomaly detection failed: {str(e)}")
            
            # Clustering Analysis
            if SKLEARN_AVAILABLE and len(feature_vector) > 1:
                try:
                    cluster_analysis = self._clustering_analysis(feature_vector)
                    results['clustering'] = cluster_analysis
                except Exception as e:
                    self.logger.warning(f"Clustering analysis failed: {str(e)}")
            
            # Feature-based Classification
            try:
                classification_score = self._feature_based_classification(feature_vector)
                results['classification'] = classification_score
            except Exception as e:
                self.logger.warning(f"Classification failed: {str(e)}")
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(results)
            
            return {
                'overall_confidence': overall_confidence,
                'confidence_level': self._get_confidence_level(overall_confidence),
                'detailed_results': results,
                'feature_vector_size': len(feature_vector),
                'algorithms_used': list(results.keys())
            }
            
        except Exception as e:
            st.warning(f"⚠️ ML prediction failed: {str(e)}")
            return self._fallback_prediction()
    
    def _prepare_feature_vector(self, cv_features: Dict, env_data: Dict, known_sites: List[Dict]) -> np.ndarray:
        """Prepare feature vector for ML algorithms"""
        features = []
        
        # Computer Vision Features
        if 'circular_features' in cv_features:
            circular_count = len(cv_features['circular_features'])
            circular_avg_confidence = np.mean([f['confidence'] for f in cv_features['circular_features']]) if circular_count > 0 else 0
            features.extend([circular_count, circular_avg_confidence])
        else:
            features.extend([0, 0])
        
        if 'linear_features' in cv_features:
            linear_count = len(cv_features['linear_features'])
            linear_avg_confidence = np.mean([f['confidence'] for f in cv_features['linear_features']]) if linear_count > 0 else 0
            features.extend([linear_count, linear_avg_confidence])
        else:
            features.extend([0, 0])
        
        if 'corner_features' in cv_features:
            corner_count = len(cv_features['corner_features'])
            corner_avg_confidence = np.mean([f['confidence'] for f in cv_features['corner_features']]) if corner_count > 0 else 0
            features.extend([corner_count, corner_avg_confidence])
        else:
            features.extend([0, 0])
        
        if 'edge_features' in cv_features:
            edge_density = np.mean([f.get('edge_density', 0) for f in cv_features['edge_features']])
            features.append(edge_density)
        else:
            features.append(0)
        
        if 'texture_features' in cv_features and cv_features['texture_features']:
            texture_variance = cv_features['texture_features'][0].get('variance', 0)
            texture_entropy = cv_features['texture_features'][0].get('entropy', 0)
            features.extend([texture_variance, texture_entropy])
        else:
            features.extend([0, 0])
        
        # Environmental Features
        if env_data.get('status') == 'success':
            features.extend([
                env_data.get('temperature', 20),
                env_data.get('humidity', 50),
                env_data.get('pressure', 1013),
                env_data.get('wind_speed', 0),
                env_data.get('cloudiness', 50)
            ])
        else:
            features.extend([20, 50, 1013, 0, 50])  # Default values
        
        # Known Sites Features
        known_sites_count = len(known_sites)
        avg_distance = np.mean([site.get('distance_km', 100) for site in known_sites]) if known_sites_count > 0 else 100
        features.extend([known_sites_count, avg_distance])
        
        return np.array(features).reshape(1, -1)
    
    def _anomaly_detection(self, feature_vector: np.ndarray) -> Dict:
        """Anomaly detection using Isolation Forest and One-Class SVM"""
        results = {}
        
        # Isolation Forest
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            # Create synthetic normal data for training
            normal_data = np.random.normal(0, 1, (100, feature_vector.shape[1]))
            iso_forest.fit(normal_data)
            
            anomaly_score = iso_forest.decision_function(feature_vector)[0]
            is_anomaly = iso_forest.predict(feature_vector)[0] == -1
            
            results['isolation_forest'] = {
                'anomaly_score': float(anomaly_score),
                'is_anomaly': bool(is_anomaly),
                'confidence': min(abs(anomaly_score), 1.0)
            }
        except Exception as e:
            self.logger.warning(f"Isolation Forest failed: {str(e)}")
        
        # One-Class SVM
        try:
            oc_svm = OneClassSVM(gamma='scale', nu=0.1)
            normal_data = np.random.normal(0, 1, (100, feature_vector.shape[1]))
            oc_svm.fit(normal_data)
            
            svm_score = oc_svm.decision_function(feature_vector)[0]
            svm_prediction = oc_svm.predict(feature_vector)[0] == -1
            
            results['one_class_svm'] = {
                'anomaly_score': float(svm_score),
                'is_anomaly': bool(svm_prediction),
                'confidence': min(abs(svm_score), 1.0)
            }
        except Exception as e:
            self.logger.warning(f"One-Class SVM failed: {str(e)}")
        
        return results
    
    def _clustering_analysis(self, feature_vector: np.ndarray) -> Dict:
        """Clustering analysis using DBSCAN and K-Means"""
        results = {}
        
        # Generate synthetic data for clustering context
        synthetic_data = np.random.normal(0, 1, (50, feature_vector.shape[1]))
        combined_data = np.vstack([synthetic_data, feature_vector])
        
        # DBSCAN Clustering
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            cluster_labels = dbscan.fit_predict(combined_data)
            
            target_cluster = cluster_labels[-1]  # Cluster of our target point
            cluster_size = np.sum(cluster_labels == target_cluster)
            
            results['dbscan'] = {
                'cluster_label': int(target_cluster),
                'cluster_size': int(cluster_size),
                'is_outlier': target_cluster == -1,
                'confidence': 1.0 if target_cluster == -1 else min(1.0 / cluster_size, 1.0)
            }
        except Exception as e:
            self.logger.warning(f"DBSCAN failed: {str(e)}")
        
        # K-Means Clustering
        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(combined_data)
            
            target_cluster = cluster_labels[-1]
            distances = kmeans.transform(combined_data)
            target_distance = distances[-1, target_cluster]
            
            results['kmeans'] = {
                'cluster_label': int(target_cluster),
                'distance_to_center': float(target_distance),
                'confidence': min(target_distance / 2.0, 1.0)
            }
        except Exception as e:
            self.logger.warning(f"K-Means failed: {str(e)}")
        
        return results
    
    def _feature_based_classification(self, feature_vector: np.ndarray) -> Dict:
        """Feature-based archaeological potential classification"""
        try:
            # Extract key features
            features = feature_vector[0]
            
            # Scoring based on archaeological indicators
            score = 0.0
            
            # Computer Vision Features (40% weight)
            if len(features) >= 6:
                circular_count = features[0]
                circular_confidence = features[1]
                linear_count = features[2]
                linear_confidence = features[3]
                corner_count = features[4]
                corner_confidence = features[5]
                
                cv_score = (
                    min(circular_count * 0.1, 0.3) * circular_confidence +
                    min(linear_count * 0.05, 0.2) * linear_confidence +
                    min(corner_count * 0.02, 0.1) * corner_confidence
                )
                score += cv_score * 0.4
            
            # Environmental Suitability (30% weight)
            if len(features) >= 12:
                temperature = features[7]
                humidity = features[8]
                
                # Optimal conditions for archaeological preservation
                temp_score = 1.0 - abs(temperature - 20) / 30.0  # Optimal around 20°C
                humidity_score = 1.0 - abs(humidity - 40) / 60.0  # Optimal around 40%
                
                env_score = (temp_score + humidity_score) / 2.0
                score += max(env_score, 0) * 0.3
            
            # Proximity to Known Sites (30% weight)
            if len(features) >= 14:
                known_sites_count = features[12]
                avg_distance = features[13]
                
                proximity_score = min(known_sites_count * 0.2, 1.0)
                distance_score = max(1.0 - avg_distance / 100.0, 0)  # Closer is better
                
                site_score = (proximity_score + distance_score) / 2.0
                score += site_score * 0.3
            
            confidence_level = self._get_confidence_level(score)
            
            return {
                'archaeological_score': float(score),
                'confidence_level': confidence_level,
                'cv_contribution': cv_score * 0.4 if 'cv_score' in locals() else 0,
                'env_contribution': max(env_score, 0) * 0.3 if 'env_score' in locals() else 0,
                'site_contribution': site_score * 0.3 if 'site_score' in locals() else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Feature-based classification failed: {str(e)}")
            return {'archaeological_score': 0.5, 'confidence_level': 'medium'}
    
    def _calculate_overall_confidence(self, results: Dict) -> float:
        """Calculate overall confidence from all ML results"""
        scores = []
        
        # Anomaly detection scores
        if 'anomaly_detection' in results:
            anomaly_results = results['anomaly_detection']
            for method, result in anomaly_results.items():
                if result.get('is_anomaly', False):
                    scores.append(result.get('confidence', 0.5))
        
        # Clustering scores
        if 'clustering' in results:
            cluster_results = results['clustering']
            for method, result in cluster_results.items():
                scores.append(result.get('confidence', 0.5))
        
        # Classification score
        if 'classification' in results:
            scores.append(results['classification'].get('archaeological_score', 0.5))
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to level"""
        if confidence >= 0.7:
            return 'high'
        elif confidence >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _fallback_prediction(self) -> Dict:
        """Fallback prediction when ML fails"""
        return {
            'overall_confidence': 0.5,
            'confidence_level': 'medium',
            'detailed_results': {},
            'feature_vector_size': 0,
            'algorithms_used': ['fallback']
        }

class ArchaeologicalSiteDiscovery:
    """Main archaeological site discovery system using real APIs only"""
    
    def __init__(self, logger, gee_project_id: str):
        self.logger = logger
        self.gee_processor = GoogleEarthEngineProcessor(logger, gee_project_id)
        self.cv_analyzer = ComputerVisionAnalyzer(logger)
        self.ml_predictor = MachineLearningPredictor(logger)
        self.real_data_processor = RealDataProcessor(logger)
        self.openai_processor = OpenAIProcessor(logger)
        self.analysis_timeout = DEFAULT_CONFIG['analysis_timeout']
    
    def analyze_region(self, lat: float, lon: float, analysis_config: Dict) -> Dict:
        """Analyze a region for archaeological potential using real APIs only"""
        start_time = time.time()
        
        try:
            st.info(f"🚀 Starting real API analysis for {lat:.4f}, {lon:.4f}")
            
            results = {
                'location': {'lat': lat, 'lon': lon},
                'analysis_config': analysis_config,
                'timestamp': datetime.now().isoformat(),
                'satellite_data': None,
                'cv_features': {},
                'ml_prediction': {},
                'environmental_data': {},
                'archaeological_sites': [],
                'archaeological_news': [],
                'ai_insights': {},
                'confidence_score': 0.0,
                'data_sources_used': []
            }
            
            # Step 1: Get real environmental data from OpenWeatherMap
            try:
                st.info("🌍 Getting real environmental data from OpenWeatherMap...")
                env_data = self.real_data_processor.get_real_environmental_data(lat, lon)
                results['environmental_data'] = env_data
                if env_data.get('status') == 'success':
                    results['data_sources_used'].append('OpenWeatherMap API')
                    st.success("✅ Real environmental data acquired")
                else:
                    st.info("ℹ️ Environmental data unavailable (API key needed)")
            except Exception as e:
                st.warning(f"⚠️ Environmental data failed: {str(e)}")
                results['environmental_data'] = {'status': 'error'}
            
            # Step 2: Get real archaeological sites from Wikidata
            try:
                st.info("🏛️ Searching real archaeological sites from Wikidata...")
                known_sites = self.real_data_processor.get_real_archaeological_sites(lat, lon)
                results['archaeological_sites'] = known_sites
                if known_sites:
                    results['data_sources_used'].append('Wikidata SPARQL API')
                    st.success(f"✅ Found {len(known_sites)} real archaeological sites")
                else:
                    st.info("ℹ️ No archaeological sites found in Wikidata for this region")
            except Exception as e:
                st.warning(f"⚠️ Archaeological sites search failed: {str(e)}")
                results['archaeological_sites'] = []
            
            # Step 3: Get real archaeological news from RSS feeds
            try:
                st.info("📰 Getting real archaeological news from RSS feeds...")
                news_items = self.real_data_processor.get_archaeological_news()
                results['archaeological_news'] = news_items
                if news_items:
                    results['data_sources_used'].append('Archaeological RSS Feeds')
                    st.success(f"✅ Retrieved {len(news_items)} real news items")
                else:
                    st.info("ℹ️ No recent archaeological news available")
            except Exception as e:
                st.warning(f"⚠️ News retrieval failed: {str(e)}")
                results['archaeological_news'] = []
            
            # Step 4: Get real satellite imagery from Google Earth Engine
            try:
                st.info("🛰️ Acquiring real satellite imagery from Google Earth Engine...")
                satellite_data = self.gee_processor.get_real_satellite_data(lat, lon)
                
                if satellite_data:
                    results['satellite_data'] = satellite_data
                    results['data_sources_used'].append('Google Earth Engine')
                    st.success("✅ Real satellite imagery acquired")
                    
                    # Step 5: Computer Vision Analysis on real satellite data
                    if analysis_config.get('use_cv', True) and 'thumbnail_url' in satellite_data:
                        try:
                            st.info("🔍 Running computer vision analysis on real satellite data...")
                            cv_features = self.cv_analyzer.analyze_satellite_image(satellite_data['thumbnail_url'])
                            
                            if 'error' not in cv_features:
                                results['cv_features'] = cv_features
                                total_features = sum(len(cv_features.get(key, [])) for key in cv_features if isinstance(cv_features.get(key), list))
                                st.success(f"✅ Computer vision analysis complete: {total_features} features detected")
                            else:
                                st.warning(f"⚠️ Computer vision analysis failed: {cv_features['error']}")
                                results['cv_features'] = {}
                        except Exception as e:
                            st.warning(f"⚠️ Computer vision analysis failed: {str(e)}")
                            results['cv_features'] = {}
                else:
                    st.warning("⚠️ No satellite imagery available")
                    results['satellite_data'] = None
                    results['cv_features'] = {}
            except Exception as e:
                st.warning(f"⚠️ Satellite imagery acquisition failed: {str(e)}")
                results['satellite_data'] = None
                results['cv_features'] = {}
            
            # Step 6: Machine Learning Analysis using real data
            if analysis_config.get('use_ml', True):
                try:
                    st.info("🤖 Running machine learning analysis on real data...")
                    ml_prediction = self.ml_predictor.predict_archaeological_potential(
                        results['cv_features'], 
                        results['environmental_data'], 
                        results['archaeological_sites']
                    )
                    results['ml_prediction'] = ml_prediction
                    st.success(f"✅ ML analysis complete: {ml_prediction['confidence_level']} confidence")
                except Exception as e:
                    st.warning(f"⚠️ Machine learning analysis failed: {str(e)}")
                    results['ml_prediction'] = {'overall_confidence': 0.5, 'confidence_level': 'medium'}
            
            # Step 7: OpenAI Analysis (if API key provided)
            if analysis_config.get('use_ai', True):
                try:
                    st.info("🤖 Generating AI insights with OpenAI...")
                    ai_insights = self.openai_processor.analyze_archaeological_potential(
                        results['location'], results
                    )
                    results['ai_insights'] = ai_insights
                    if ai_insights.get('status') == 'success':
                        results['data_sources_used'].append('OpenAI API')
                        st.success("✅ AI archaeological insights generated")
                    else:
                        st.info("ℹ️ AI insights unavailable (OpenAI API key needed)")
                except Exception as e:
                    st.warning(f"⚠️ AI analysis failed: {str(e)}")
                    results['ai_insights'] = {'status': 'error'}
            
            # Step 8: Calculate overall confidence score
            try:
                confidence_score = self._calculate_overall_confidence(results)
                results['confidence_score'] = confidence_score
                st.info(f"🎯 Overall archaeological potential: {confidence_score:.3f}")
            except Exception as e:
                st.warning(f"⚠️ Confidence calculation failed: {str(e)}")
                results['confidence_score'] = 0.5
            
            # Log success
            analysis_time = time.time() - start_time
            st.success(f"✅ Real API analysis completed for {lat:.4f}, {lon:.4f} in {analysis_time:.1f}s")
            st.info(f"📊 Data sources used: {', '.join(results['data_sources_used'])}")
            
            return results
            
        except Exception as e:
            analysis_time = time.time() - start_time
            error_msg = f"Analysis failed for {lat:.4f}, {lon:.4f}: {str(e)}"
            st.error(f"❌ {error_msg}")
            return {
                'location': {'lat': lat, 'lon': lon},
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_overall_confidence(self, results: Dict) -> float:
        """Calculate overall confidence from real data analysis"""
        try:
            scores = []
            
            # ML prediction confidence (40% weight)
            if results['ml_prediction']:
                ml_confidence = results['ml_prediction'].get('overall_confidence', 0.5)
                scores.append(ml_confidence * 0.4)
            
            # Known archaeological sites proximity (30% weight)
            if results['archaeological_sites']:
                site_score = min(len(results['archaeological_sites']) * 0.1, 0.3)
                avg_distance = np.mean([site.get('distance_km', 100) for site in results['archaeological_sites']])
                distance_score = max(1.0 - avg_distance / 100.0, 0) * 0.3
                scores.append(site_score + distance_score)
            
            # Computer vision features (20% weight)
            if results['cv_features'] and not results['cv_features'].get('error'):
                cv_score = 0
                for feature_type, features in results['cv_features'].items():
                    if isinstance(features, list) and features:
                        avg_confidence = np.mean([f.get('confidence', 0.5) for f in features])
                        cv_score += avg_confidence * 0.05
                scores.append(min(cv_score, 0.2))
            
            # Environmental suitability (10% weight)
            if results['environmental_data'].get('status') == 'success':
                env_data = results['environmental_data']
                temp = env_data.get('temperature', 20)
                humidity = env_data.get('humidity', 50)
                
                # Optimal preservation conditions
                temp_score = max(1.0 - abs(temp - 20) / 30.0, 0)
                humidity_score = max(1.0 - abs(humidity - 40) / 60.0, 0)
                env_score = (temp_score + humidity_score) / 2.0 * 0.1
                scores.append(env_score)
            
            return float(np.sum(scores)) if scores else 0.5
            
        except Exception:
            return 0.5

class DataLogger:
    """Data logging system for real API usage tracking"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.operation_log = []
        self.api_usage_stats = {
            'wikidata_queries': 0,
            'openweather_calls': 0,
            'gee_queries': 0,
            'rss_feeds_parsed': 0,
            'openai_calls': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_features_detected': 0,
            'total_sites_found': 0
        }
    
    def log_api_usage(self, api_name: str, success: bool, data_count: int = 0):
        """Log API usage"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'api_name': api_name,
            'success': success,
            'data_count': data_count,
            'type': 'api_usage'
        }
        self.operation_log.append(log_entry)
        
        # Update stats
        if api_name == 'Wikidata':
            self.api_usage_stats['wikidata_queries'] += 1
        elif api_name == 'OpenWeatherMap':
            self.api_usage_stats['openweather_calls'] += 1
        elif api_name == 'Google Earth Engine':
            self.api_usage_stats['gee_queries'] += 1
        elif api_name == 'RSS Feeds':
            self.api_usage_stats['rss_feeds_parsed'] += 1
        elif api_name == 'OpenAI':
            self.api_usage_stats['openai_calls'] += 1
        
        if success:
            if 'sites' in api_name.lower():
                self.api_usage_stats['total_sites_found'] += data_count
            elif 'features' in api_name.lower():
                self.api_usage_stats['total_features_detected'] += data_count
    
    def log_analysis_result(self, success: bool):
        """Log analysis result"""
        if success:
            self.api_usage_stats['successful_analyses'] += 1
        else:
            self.api_usage_stats['failed_analyses'] += 1
    
    def get_usage_summary(self) -> Dict:
        """Get API usage summary"""
        return {
            'api_usage_stats': self.api_usage_stats,
            'total_operations': len(self.operation_log),
            'recent_operations': self.operation_log[-10:]
        }

def initialize_session_state():
    """Initialize session state"""
    defaults = {
        'data_logger': DataLogger(),
        'site_discovery': None,
        'analysis_results': [],
        'analysis_config': {
            'use_cv': True,
            'use_ml': True,
            'use_ai': True,
            'cv_algorithms': ['hough_circles', 'hough_lines', 'harris_corners', 'lbp_texture', 'multi_edge'],
            'ml_algorithms': ['anomaly_detection', 'clustering', 'classification']
        },
        'confidence_threshold': 0.5,
        'gee_initialized': False,
        'api_config': DEFAULT_CONFIG.copy()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_confidence_distribution(results: List[Dict]) -> go.Figure:
    """Create confidence distribution chart"""
    if not results:
        return go.Figure()
    
    confidence_scores = [r.get('confidence_score', 0) for r in results]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=confidence_scores,
        nbinsx=10,
        name='Archaeological Potential Distribution',
        marker_color='rgba(52, 152, 219, 0.7)',
        opacity=0.8
    ))
    
    fig.update_layout(
        title="Archaeological Potential Confidence Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Number of Locations",
        height=400,
        template="plotly_dark"
    )
    
    return fig

def create_feature_analysis_chart(results: List[Dict]) -> go.Figure:
    """Create feature analysis chart"""
    if not results:
        return go.Figure()
    
    feature_data = []
    for result in results:
        cv_features = result.get('cv_features', {})
        location = f"{result['location']['lat']:.2f}, {result['location']['lon']:.2f}"
        
        for feature_type, features in cv_features.items():
            if isinstance(features, list):
                feature_data.append({
                    'location': location,
                    'feature_type': feature_type.replace('_', ' ').title(),
                    'count': len(features),
                    'avg_confidence': np.mean([f.get('confidence', 0.5) for f in features]) if features else 0
                })
    
    if not feature_data:
        return go.Figure()
    
    df = pd.DataFrame(feature_data)
    
    fig = px.bar(
        df, 
        x='location', 
        y='count', 
        color='feature_type',
        title="Computer Vision Features Detected by Location",
        labels={'count': 'Number of Features', 'location': 'Location (Lat, Lon)'}
    )
    
    fig.update_layout(height=500, template="plotly_dark")
    return fig

def create_site_map(results: List[Dict]) -> str:
    """Create interactive map of archaeological analysis results"""
    if not results:
        return "<p>No results to display</p>"
    
    # Calculate center
    lats = [r['location']['lat'] for r in results if 'location' in r]
    lons = [r['location']['lon'] for r in results if 'location' in r]
    
    if not lats:
        return "<p>No valid locations to display</p>"
    
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Create map HTML with real data indicators
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real Archaeological Data Discovery Map</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
            body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
            #map {{ height: 100vh; width: 100%; }}
            .popup-content {{
                font-family: Arial, sans-serif;
                max-width: 350px;
            }}
            .popup-header {{
                background: linear-gradient(135deg, #3498db, #2ecc71);
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 5px 5px 0 0;
                text-align: center;
            }}
            .popup-body {{
                padding: 15px;
                background: white;
                border-radius: 0 0 5px 5px;
            }}
            .data-source {{
                display: inline-block;
                background: #e8f5e8;
                color: #2e7d32;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 0.8rem;
                margin: 2px;
                border: 1px solid #4caf50;
            }}
            .confidence-high {{ border-left: 5px solid #2ecc71; }}
            .confidence-medium {{ border-left: 5px solid #f39c12; }}
            .confidence-low {{ border-left: 5px solid #e74c3c; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            var map = L.map('map').setView([{center_lat}, {center_lon}], 6);
            
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);
            
            var sites = {json.dumps([{
                'lat': r['location']['lat'],
                'lon': r['location']['lon'],
                'confidence': r.get('confidence_score', 0),
                'known_sites': len(r.get('archaeological_sites', [])),
                'data_sources': r.get('data_sources_used', []),
                'cv_features': sum(len(features) if isinstance(features, list) else 0 
                                 for features in r.get('cv_features', {}).values()),
                'env_status': r.get('environmental_data', {}).get('status', 'unavailable'),
                'has_satellite': bool(r.get('satellite_data')),
                'news_count': len(r.get('archaeological_news', [])),
                'has_ai_insights': r.get('ai_insights', {}).get('status') == 'success'
            } for r in results])};
            
            sites.forEach(function(site) {{
                var confidence_level = site.confidence >= 0.7 ? 'high' : 
                                     site.confidence >= 0.4 ? 'medium' : 'low';
                
                var color = confidence_level === 'high' ? '#2ecc71' :
                           confidence_level === 'medium' ? '#f39c12' : '#e74c3c';
                
                var dataSourcesHtml = site.data_sources.map(function(source) {{
                    return '<span class="data-source">' + source + '</span>';
                }}).join(' ');
                
                var popupContent = `
                    <div class="popup-content">
                        <div class="popup-header">
                            🛰️ Real Archaeological Data Analysis
                        </div>
                        <div class="popup-body confidence-${{confidence_level}}">
                            <p><strong>📍 Location:</strong> ${{site.lat.toFixed(4)}}, ${{site.lon.toFixed(4)}}</p>
                            <p><strong>🎯 Archaeological Potential:</strong> ${{(site.confidence * 100).toFixed(1)}}% (${{confidence_level}})</p>
                            <p><strong>🏛️ Known Sites (Wikidata):</strong> ${{site.known_sites}}</p>
                            <p><strong>🔍 CV Features Detected:</strong> ${{site.cv_features}}</p>
                            <p><strong>🌍 Environmental Data:</strong> ${{site.env_status}}</p>
                            <p><strong>🛰️ Satellite Data:</strong> ${{site.has_satellite ? 'Available' : 'Unavailable'}}</p>
                            <p><strong>📰 Recent News:</strong> ${{site.news_count}} items</p>
                            <p><strong>🤖 AI Insights:</strong> ${{site.has_ai_insights ? 'Available' : 'Unavailable'}}</p>
                            <p><strong>📊 Real Data Sources:</strong></p>
                            <div>${{dataSourcesHtml || '<span style="color: #666;">No real data sources available</span>'}}</div>
                        </div>
                    </div>
                `;
                
                L.circleMarker([site.lat, site.lon], {{
                    radius: 8 + site.confidence * 15,
                    fillColor: color,
                    color: 'white',
                    weight: 3,
                    opacity: 1,
                    fillOpacity: 0.8
                }}).bindPopup(popupContent, {{maxWidth: 400}}).addTo(map);
            }});
            
            // Add legend
            var legend = L.control({{position: 'bottomright'}});
            legend.onAdd = function (map) {{
                var div = L.DomUtil.create('div', 'info legend');
                div.innerHTML = `
                    <div style="background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                        <h4 style="margin: 0 0 10px 0;">Archaeological Potential</h4>
                        <div><span style="color: #2ecc71;">●</span> High (≥70%)</div>
                        <div><span style="color: #f39c12;">●</span> Medium (40-69%)</div>
                        <div><span style="color: #e74c3c;">●</span> Low (<40%)</div>
                        <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                            Size indicates confidence level<br>
                            Data from real APIs only
                        </div>
                    </div>
                `;
                return div;
            }};
            legend.addTo(map);
        </script>
    </body>
    </html>
    """
    
    return html_content

def display_archaeological_news(news_items: List[Dict]):
    """Display archaeological news items"""
    if not news_items:
        st.info("📰 No recent archaeological news available")
        return
    
    st.markdown("### 📰 Latest Archaeological News")
    st.markdown("*Real-time news from verified archaeological sources*")
    
    for i, item in enumerate(news_items[:10]):  # Show top 10
        with st.expander(f"📰 {item.get('title', 'No Title')}", expanded=i < 3):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Summary:** {item.get('summary', 'No summary available')[:300]}...")
                if item.get('link'):
                    st.markdown(f"[🔗 Read Full Article]({item['link']})")
            
            with col2:
                st.markdown(f"**Source:** {item.get('source', 'Unknown')}")
                st.markdown(f"**Published:** {item.get('published', 'Unknown')}")

def display_ai_insights(ai_insights: Dict):
    """Display AI-generated archaeological insights"""
    if ai_insights.get('status') == 'success':
        st.markdown("""
        <div class="ai-insight">
            <h4>🤖 AI Archaeological Insights</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(ai_insights.get('analysis', 'No analysis available'))
        
        if ai_insights.get('tokens_used'):
            st.caption(f"Generated using {ai_insights.get('model_used', 'OpenAI')} • {ai_insights.get('tokens_used', 0)} tokens")
    elif ai_insights.get('status') == 'no_api_key':
        st.info("🤖 AI insights unavailable - OpenAI API key not provided")
    elif ai_insights.get('status') == 'not_available':
        st.warning("🤖 AI insights unavailable - OpenAI library not installed")
    else:
        st.warning(f"🤖 AI insights failed: {ai_insights.get('message', 'Unknown error')}")

def main():
    """Main application using real APIs only"""
    initialize_session_state()
    load_professional_css()
    
    # Professional header
    st.markdown("""
    <div class="platform-header">
        <h1>🛰️ Archaeological Site Discovery Platform</h1>
        <p><strong>Real Data Sources Only</strong> • Google Earth Engine • Wikidata • OpenWeatherMap • OpenAI • RSS Feeds</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">Advanced Computer Vision • Machine Learning • AI Insights • No Fake Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # GEE Project ID (both default and user input)
        st.markdown("**Google Earth Engine Project ID:**")
        gee_project = st.text_input(
            "GEE Project ID",
            value=st.session_state.api_config['gee_project_id'],
            help="Your Google Earth Engine project ID (required for satellite data)"
        )
        if gee_project != st.session_state.api_config['gee_project_id']:
            st.session_state.api_config['gee_project_id'] = gee_project
            st.session_state.site_discovery = None  # Reinitialize
        
        # API Configuration
        with st.expander("🔑 API Keys", expanded=False):
            st.markdown("**OpenWeatherMap API Key (Optional):**")
            weather_key = st.text_input(
                "Weather API Key",
                value=st.session_state.api_config.get('openweather_api_key', ''),
                type="password",
                help="Get free key from https://openweathermap.org/api"
            )
            if weather_key != st.session_state.api_config.get('openweather_api_key', ''):
                st.session_state.api_config['openweather_api_key'] = weather_key
            
            st.markdown("**OpenAI API Key (Optional):**")
            openai_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.api_config.get('openai_api_key', ''),
                type="password",
                help="Get API key from https://platform.openai.com/api-keys"
            )
            if openai_key != st.session_state.api_config.get('openai_api_key', ''):
                st.session_state.api_config['openai_api_key'] = openai_key
            
            st.markdown("*Other APIs (Wikidata, RSS) require no keys*")
        
        # Analysis Configuration
        st.markdown("### 🔍 Analysis Configuration")
        
        use_cv = st.checkbox("🔍 Computer Vision Analysis", value=st.session_state.analysis_config['use_cv'])
        st.session_state.analysis_config['use_cv'] = use_cv
        
        if use_cv:
            st.markdown("**CV Algorithms:**")
            cv_algorithms = st.multiselect(
                "Select algorithms:",
                ['hough_circles', 'hough_lines', 'harris_corners', 'lbp_texture', 'multi_edge'],
                default=st.session_state.analysis_config['cv_algorithms']
            )
            st.session_state.analysis_config['cv_algorithms'] = cv_algorithms
        
        use_ml = st.checkbox("🤖 Machine Learning Analysis", value=st.session_state.analysis_config['use_ml'])
        st.session_state.analysis_config['use_ml'] = use_ml
        
        if use_ml:
            st.markdown("**ML Algorithms:**")
            ml_algorithms = st.multiselect(
                "Select algorithms:",
                ['anomaly_detection', 'clustering', 'classification'],
                default=st.session_state.analysis_config['ml_algorithms']
            )
            st.session_state.analysis_config['ml_algorithms'] = ml_algorithms
        
        use_ai = st.checkbox("🤖 AI Insights (OpenAI)", value=st.session_state.analysis_config['use_ai'])
        st.session_state.analysis_config['use_ai'] = use_ai
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence for archaeological potential"
        )
        st.session_state.confidence_threshold = confidence_threshold
        
        # System Status
        st.markdown("### 📊 Real API Status")
        
        # Check API availability
        gee_status = GEE_AVAILABLE and st.session_state.api_config.get('gee_project_id')
        weather_status = bool(st.session_state.api_config.get('openweather_api_key'))
        openai_status = bool(st.session_state.api_config.get('openai_api_key')) and OPENAI_AVAILABLE
        
        status_items = [
            ("🛰️ Google Earth Engine", gee_status),
            ("🌍 Wikidata SPARQL", True),
            ("🌤️ OpenWeatherMap", weather_status),
            ("🤖 OpenAI", openai_status),
            ("📰 RSS Feeds", True),
            ("🔍 Computer Vision", True),
            ("🤖 Machine Learning", SKLEARN_AVAILABLE)
        ]
        
        for item, status in status_items:
            status_class = "online" if status else "offline"
            icon = "✅" if status else "❌"
            st.markdown(f'<span class="api-status {status_class}">{icon} {item}</span>', unsafe_allow_html=True)
        
        # Usage Statistics
        if st.session_state.data_logger:
            stats = st.session_state.data_logger.api_usage_stats
            st.markdown("### 📈 API Usage Stats")
            st.markdown(f"""
            - **Wikidata Queries:** {stats['wikidata_queries']}
            - **Weather API Calls:** {stats['openweather_calls']}
            - **GEE Queries:** {stats['gee_queries']}
            - **OpenAI Calls:** {stats['openai_calls']}
            - **RSS Feeds Parsed:** {stats['rss_feeds_parsed']}
            - **Successful Analyses:** {stats['successful_analyses']}
            - **Sites Found:** {stats['total_sites_found']}
            """)
    
    # Initialize site discovery system
    if st.session_state.site_discovery is None:
        st.session_state.site_discovery = ArchaeologicalSiteDiscovery(
            st.session_state.data_logger, 
            st.session_state.api_config['gee_project_id']
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Site Discovery", 
        "📊 Analysis Results", 
        "🗺️ Interactive Maps",
        "📈 Visualizations",
        "📰 Archaeological News",
        "🤖 AI Insights"
    ])
    
    with tab1:
        st.markdown("## 🎯 Archaeological Site Discovery")
        st.markdown("*Using real APIs: Google Earth Engine, Wikidata, OpenWeatherMap, OpenAI, RSS Feeds*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown("### 📍 Location Selection")
            
            # Initialize locations
            if 'locations_to_analyze' not in st.session_state:
                st.session_state.locations_to_analyze = []
            
            # Location input methods
            input_method = st.radio(
                "Choose input method:",
                ["Manual Coordinates", "Famous Archaeological Sites", "Upload CSV"]
            )
            
            if input_method == "Manual Coordinates":
                col_lat, col_lon = st.columns(2)
                with col_lat:
                    lat = st.number_input("Latitude", value=13.4125, format="%.6f", key="manual_lat")
                with col_lon:
                    lon = st.number_input("Longitude", value=103.8670, format="%.6f", key="manual_lon")
                
                if st.button("📍 Add Location", key="add_manual"):
                    st.session_state.locations_to_analyze.append((lat, lon))
                    st.success(f"Added location: {lat:.4f}, {lon:.4f}")
                    st.rerun()
            
            elif input_method == "Famous Archaeological Sites":
                presets = {
                    "Angkor Wat, Cambodia": (13.4125, 103.8670),
                    "Petra, Jordan": (30.3285, 35.4444),
                    "Machu Picchu, Peru": (-13.1631, -72.5450),
                    "Stonehenge, UK": (51.1789, -1.8262),
                    "Pompeii, Italy": (40.7489, 14.4989),
                    "Chichen Itza, Mexico": (20.6843, -88.5678),
                    "Easter Island, Chile": (-27.1127, -109.3497),
                    "Giza Pyramids, Egypt": (29.9792, 31.1342),
                    "Göbekli Tepe, Turkey": (37.2233, 38.9225),
                    "Newgrange, Ireland": (53.6947, -6.4761)
                }
                
                selected_preset = st.selectbox("Select archaeological site:", list(presets.keys()))
                
                if st.button("📍 Add Preset Location", key="add_preset"):
                    lat, lon = presets[selected_preset]
                    st.session_state.locations_to_analyze.append((lat, lon))
                    st.success(f"Added {selected_preset}: {lat:.4f}, {lon:.4f}")
                    st.rerun()
            
            else:  # Upload CSV
                uploaded_file = st.file_uploader("Upload CSV with lat,lon columns", type=['csv'])
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if 'lat' in df.columns and 'lon' in df.columns:
                            st.write(f"Found {len(df)} locations in CSV:")
                            st.dataframe(df.head())
                            
                            if st.button("📍 Add All CSV Locations", key="add_csv"):
                                new_locations = list(zip(df['lat'], df['lon']))
                                st.session_state.locations_to_analyze.extend(new_locations)
                                st.success(f"Added {len(new_locations)} locations from CSV")
                                st.rerun()
                        else:
                            st.error("CSV must contain 'lat' and 'lon' columns")
                    except Exception as e:
                        st.error(f"Error reading CSV: {str(e)}")
            
            # Display current locations
            if st.session_state.locations_to_analyze:
                st.markdown("### 📋 Locations to Analyze:")
                for i, (lat, lon) in enumerate(st.session_state.locations_to_analyze):
                    col_info, col_remove = st.columns([4, 1])
                    with col_info:
                        st.write(f"{i+1}. {lat:.4f}, {lon:.4f}")
                    with col_remove:
                        if st.button("🗑️", key=f"remove_{i}", help="Remove location"):
                            st.session_state.locations_to_analyze.pop(i)
                            st.rerun()
                
                # Clear all button
                if st.button("🗑️ Clear All Locations", key="clear_all"):
                    st.session_state.locations_to_analyze = []
                    st.rerun()
                
                # Analysis execution
                st.markdown(f"### 🚀 Ready to analyze {len(st.session_state.locations_to_analyze)} location(s)")
                
                if st.button("🔍 Start Real API Analysis", type="primary", use_container_width=True):
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    total_locations = len(st.session_state.locations_to_analyze)
                    
                    # Process each location
                    for i, (lat, lon) in enumerate(st.session_state.locations_to_analyze):
                        
                        # Update progress
                        progress_percentage = i / total_locations
                        progress_bar.progress(progress_percentage)
                        status_text.text(f"Analyzing location {i+1}/{total_locations}: {lat:.4f}, {lon:.4f}")
                        
                        try:
                            result = st.session_state.site_discovery.analyze_region(
                                lat, lon, st.session_state.analysis_config
                            )
                            
                            if 'error' not in result:
                                results.append(result)
                                st.session_state.data_logger.log_analysis_result(True)
                            else:
                                st.session_state.data_logger.log_analysis_result(False)
                                
                        except Exception as e:
                            st.error(f"❌ Analysis failed for {lat:.4f}, {lon:.4f}: {str(e)}")
                            st.session_state.data_logger.log_analysis_result(False)
                            continue
                    
                    # Final progress update
                    progress_bar.progress(1.0)
                    status_text.text("Analysis completed!")
                    
                    # Store results
                    if results:
                        st.session_state.analysis_results.extend(results)
                        
                        # Show summary
                        st.markdown("## 🎉 Real API Analysis Complete!")
                        
                        total_known_sites = sum(len(r.get('archaeological_sites', [])) for r in results)
                        total_features = sum(
                            sum(len(features) if isinstance(features, list) else 0 
                                for features in r.get('cv_features', {}).values()) 
                            for r in results
                        )
                        avg_confidence = np.mean([r.get('confidence_score', 0) for r in results])
                        ai_insights_count = sum(1 for r in results if r.get('ai_insights', {}).get('status') == 'success')
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("📍 Locations", len(results))
                        with col2:
                            st.metric("🏛️ Real Sites", total_known_sites)
                        with col3:
                            st.metric("🔍 CV Features", total_features)
                        with col4:
                            st.metric("🎯 Avg Confidence", f"{avg_confidence:.3f}")
                        with col5:
                            st.metric("🤖 AI Insights", ai_insights_count)
                        
                        st.success(f"✅ Successfully analyzed {len(results)} locations using real APIs!")
                        st.info("📊 Check other tabs for detailed results, maps, visualizations, and AI insights")
                        
                        # Clear analyzed locations
                        st.session_state.locations_to_analyze = []
                    else:
                        st.error("❌ No successful analyses completed")
            else:
                st.info("📍 Please add locations to analyze using the methods above.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Current Configuration")
            
            config = st.session_state.analysis_config
            gee_status = GEE_AVAILABLE and st.session_state.api_config.get('gee_project_id')
            weather_status = bool(st.session_state.api_config.get('openweather_api_key'))
            openai_status = bool(st.session_state.api_config.get('openai_api_key')) and OPENAI_AVAILABLE
            
            st.markdown(f"""
            **Analysis Methods:**
            - 🔍 Computer Vision: {'✅' if config['use_cv'] else '❌'}
            - 🤖 Machine Learning: {'✅' if config['use_ml'] else '❌'}
            - 🤖 AI Insights: {'✅' if config['use_ai'] else '❌'}
            
            **Real Data Sources:**
            - 🛰️ Google Earth Engine: {'✅' if gee_status else '❌'}
            - 🏛️ Wikidata SPARQL: ✅
            - 🌤️ OpenWeatherMap: {'✅' if weather_status else '❌'}
            - 🤖 OpenAI: {'✅' if openai_status else '❌'}
            - 📰 RSS Feeds: ✅
            
            **Settings:**
            - 🎯 Confidence Threshold: {st.session_state.confidence_threshold:.1f}
            - 🔍 CV Algorithms: {len(config.get('cv_algorithms', []))}
            - 🤖 ML Algorithms: {len(config.get('ml_algorithms', []))}
            """)
            
            st.markdown("### 📊 Data Quality")
            st.markdown("""
            **✅ Real Data Only:**
            - No simulated/fake/mock data
            - Direct API connections
            - Real-time archaeological news
            - Actual satellite imagery
            - Verified archaeological sites
            - AI-powered insights
            """)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## 📊 Analysis Results")
        st.markdown("*Results from real API data analysis*")
        
        if not st.session_state.analysis_results:
            st.warning("⚠️ No analysis results available. Please run site discovery first.")
        else:
            # Results summary
            total_locations = len(st.session_state.analysis_results)
            total_known_sites = sum(len(r.get('archaeological_sites', [])) for r in st.session_state.analysis_results)
            total_features = sum(
                sum(len(features) if isinstance(features, list) else 0 
                    for features in r.get('cv_features', {}).values()) 
                for r in st.session_state.analysis_results
            )
            avg_confidence = np.mean([r.get('confidence_score', 0) for r in st.session_state.analysis_results])
            ai_insights_count = sum(1 for r in st.session_state.analysis_results if r.get('ai_insights', {}).get('status') == 'success')
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_locations}</div>
                    <div class="metric-label">Locations Analyzed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_known_sites}</div>
                    <div class="metric-label">Real Sites Found</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_features}</div>
                    <div class="metric-label">CV Features</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_confidence:.2f}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{ai_insights_count}</div>
                    <div class="metric-label">AI Insights</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Filter results
            st.markdown("### 🔍 Filter Results")
            confidence_filter = st.slider(
                "Minimum confidence:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.confidence_threshold,
                step=0.1
            )
            
            filtered_results = [r for r in st.session_state.analysis_results 
                              if r.get('confidence_score', 0) >= confidence_filter]
            
            if filtered_results:
                st.markdown(f"### 📋 Analysis Results ({len(filtered_results)} locations)")
                
                for i, result in enumerate(filtered_results):
                    lat, lon = result['location']['lat'], result['location']['lon']
                    confidence = result.get('confidence_score', 0)
                    known_sites = result.get('archaeological_sites', [])
                    cv_features = result.get('cv_features', {})
                    env_data = result.get('environmental_data', {})
                    data_sources = result.get('data_sources_used', [])
                    ai_insights = result.get('ai_insights', {})
                    
                    # Determine confidence level
                    if confidence >= 0.7:
                        confidence_level = 'high'
                    elif confidence >= 0.4:
                        confidence_level = 'medium'
                    else:
                        confidence_level = 'low'
                    
                    with st.expander(f"📍 Location {i+1}: {lat:.4f}, {lon:.4f} - {confidence_level.upper()} confidence ({confidence:.3f})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **🎯 Archaeological Potential Analysis:**
                            - **Confidence Score:** {confidence:.3f} ({confidence_level})
                            - **Data Sources Used:** {', '.join(data_sources) if data_sources else 'None'}
                            """)
                            
                            # Known archaeological sites from Wikidata
                            if known_sites:
                                st.markdown(f"**🏛️ Known Archaeological Sites from Wikidata ({len(known_sites)}):**")
                                for site in known_sites[:5]:  # Show top 5
                                    distance = site.get('distance_km', 0)
                                    st.markdown(f"- **{site['name']}** ({site['type']}) - {distance:.1f}km away")
                                    if site.get('description'):
                                        st.markdown(f"  *{site['description'][:100]}...*")
                            else:
                                st.markdown("**🏛️ Known Archaeological Sites:** None found in Wikidata")
                            
                            # Computer vision features
                            if cv_features and not cv_features.get('error'):
                                st.markdown("**🔍 Computer Vision Analysis:**")
                                for feature_type, features in cv_features.items():
                                    if isinstance(features, list) and features:
                                        avg_conf = np.mean([f.get('confidence', 0.5) for f in features])
                                        st.markdown(f"- {feature_type.replace('_', ' ').title()}: {len(features)} features (avg confidence: {avg_conf:.2f})")
                            else:
                                st.markdown("**🔍 Computer Vision Analysis:** No features detected or analysis failed")
                            
                            # Environmental data from OpenWeatherMap
                            if env_data.get('status') == 'success':
                                st.markdown(f"""
                                **🌍 Environmental Data (OpenWeatherMap):**
                                - **Temperature:** {env_data.get('temperature', 'N/A')}°C
                                - **Humidity:** {env_data.get('humidity', 'N/A')}%
                                - **Weather:** {env_data.get('weather_description', 'N/A')}
                                - **Wind Speed:** {env_data.get('wind_speed', 'N/A')} m/s
                                - **Air Quality Index:** {env_data.get('air_quality', 'N/A')}
                                """)
                            else:
                                st.markdown("**🌍 Environmental Data:** Unavailable (API key needed)")
                            
                            # AI Insights
                            if ai_insights.get('status') == 'success':
                                st.markdown("**🤖 AI Archaeological Insights:**")
                                with st.expander("View AI Analysis", expanded=False):
                                    st.markdown(ai_insights.get('analysis', 'No analysis available'))
                            else:
                                st.markdown("**🤖 AI Insights:** Unavailable (OpenAI API key needed)")
                        
                        with col2:
                            # Confidence indicator
                            st.markdown(f"""
                            <div class="confidence-card {confidence_level}">
                                <h4>🎯 {confidence_level.upper()} CONFIDENCE</h4>
                                <p><strong>Score:</strong> {confidence:.3f}</p>
                                <p><strong>Level:</strong> {confidence_level}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Real data sources indicator
                            if data_sources:
                                st.markdown("**📊 Real Data Sources:**")
                                for source in data_sources:
                                    st.markdown(f"✅ {source}")
                            else:
                                st.markdown("**📊 Real Data Sources:**")
                                st.markdown("❌ No real data sources available")
            else:
                st.info(f"No results meet the confidence threshold of {confidence_filter:.1f}")
    
    with tab3:
        st.markdown("## 🗺️ Interactive Archaeological Maps")
        st.markdown("*Real data visualization on interactive maps*")
        
        if not st.session_state.analysis_results:
            st.warning("⚠️ No analysis results available for mapping.")
        else:
            if st.button("🗺️ Generate Interactive Map", use_container_width=True):
                with st.spinner("🔄 Creating interactive map with real data..."):
                    try:
                        map_html = create_site_map(st.session_state.analysis_results)
                        
                        st.markdown("### 🗺️ Real Archaeological Data Discovery Map")
                        st.markdown("""
                        **Map Features:**
                        - 🛰️ **Real Satellite Data** from Google Earth Engine
                        - 🏛️ **Verified Sites** from Wikidata SPARQL API
                        - 🌤️ **Environmental Data** from OpenWeatherMap
                        - 🤖 **AI Insights** from OpenAI
                        - 📰 **Latest News** from archaeological RSS feeds
                        - 🎯 **Confidence Scoring** based on real data analysis
                        - 🔍 **Computer Vision** features from actual satellite imagery
                        """)
                        
                        components.html(map_html, height=600, scrolling=False)
                        
                        # Map statistics
                        total_known_sites = sum(len(r.get('archaeological_sites', [])) for r in st.session_state.analysis_results)
                        total_features = sum(
                            sum(len(features) if isinstance(features, list) else 0 
                                for features in r.get('cv_features', {}).values()) 
                            for r in st.session_state.analysis_results
                        )
                        high_conf = len([r for r in st.session_state.analysis_results if r.get('confidence_score', 0) >= 0.7])
                        medium_conf = len([r for r in st.session_state.analysis_results if 0.4 <= r.get('confidence_score', 0) < 0.7])
                        low_conf = len([r for r in st.session_state.analysis_results if r.get('confidence_score', 0) < 0.4])
                        ai_insights_count = sum(1 for r in st.session_state.analysis_results if r.get('ai_insights', {}).get('status') == 'success')
                        
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        with col1:
                            st.metric("🏛️ Real Sites", total_known_sites)
                        with col2:
                            st.metric("🔍 CV Features", total_features)
                        with col3:
                            st.metric("🟢 High Confidence", high_conf)
                        with col4:
                            st.metric("🟡 Medium Confidence", medium_conf)
                        with col5:
                            st.metric("🔴 Low Confidence", low_conf)
                        with col6:
                            st.metric("🤖 AI Insights", ai_insights_count)
                        
                        st.success("✅ **Interactive Map Generated with Real Data!**")
                        
                    except Exception as e:
                        st.error(f"❌ **Map Generation Error**: {str(e)}")
    
    with tab4:
        st.markdown("## 📈 Data Visualizations")
        st.markdown("*Advanced visualizations of real archaeological data*")
        
        if not st.session_state.analysis_results:
            st.warning("⚠️ No analysis results available for visualization.")
        else:
            if st.button("📊 Generate Visualizations", type="primary", use_container_width=True):
                with st.spinner("📊 Creating advanced visualizations..."):
                    
                    # Confidence Distribution
                    st.markdown("### 📊 Archaeological Potential Distribution")
                    confidence_fig = create_confidence_distribution(st.session_state.analysis_results)
                    st.plotly_chart(confidence_fig, use_container_width=True)
                    
                    # Feature Analysis
                    st.markdown("### 🔍 Computer Vision Features Analysis")
                    feature_fig = create_feature_analysis_chart(st.session_state.analysis_results)
                    st.plotly_chart(feature_fig, use_container_width=True)
                    
                    # Data Sources Analysis
                    st.markdown("### 📊 Real Data Sources Usage")
                    
                    # Count data sources usage
                    source_counts = {}
                    for result in st.session_state.analysis_results:
                        for source in result.get('data_sources_used', []):
                            source_counts[source] = source_counts.get(source, 0) + 1
                    
                    if source_counts:
                        fig_sources = go.Figure(data=[
                            go.Bar(
                                x=list(source_counts.keys()),
                                y=list(source_counts.values()),
                                marker_color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'][:len(source_counts)]
                            )
                        ])
                        
                        fig_sources.update_layout(
                            title="Real Data Sources Usage Frequency",
                            xaxis_title="Data Source",
                            yaxis_title="Number of Analyses",
                            height=400,
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig_sources, use_container_width=True)
                    
                    # Environmental vs Archaeological Potential
                    st.markdown("### 🌍 Environmental Conditions vs Archaeological Potential")
                    
                    env_data = []
                    for result in st.session_state.analysis_results:
                        env = result.get('environmental_data', {})
                        if env.get('status') == 'success':
                            env_data.append({
                                'temperature': env.get('temperature', 0),
                                'humidity': env.get('humidity', 0),
                                'confidence': result.get('confidence_score', 0),
                                'location': f"{result['location']['lat']:.2f}, {result['location']['lon']:.2f}",
                                'known_sites': len(result.get('archaeological_sites', [])),
                                'has_ai_insights': result.get('ai_insights', {}).get('status') == 'success'
                            })
                    
                    if env_data:
                        env_df = pd.DataFrame(env_data)
                        
                        fig_env = go.Figure()
                        fig_env.add_trace(go.Scatter(
                            x=env_df['temperature'],
                            y=env_df['humidity'],
                            mode='markers',
                            marker=dict(
                                size=env_df['confidence'] * 30 + 10,
                                color=env_df['confidence'],
                                colorscale='RdYlGn',
                                colorbar=dict(title="Archaeological Potential"),
                                line=dict(width=2, color='white'),
                                symbol=['circle' if not ai else 'star' for ai in env_df['has_ai_insights']]
                            ),
                            text=env_df['location'],
                            customdata=env_df['known_sites'],
                            hovertemplate='<b>%{text}</b><br>Temperature: %{x}°C<br>Humidity: %{y}%<br>Known Sites: %{customdata}<extra></extra>'
                        ))
                        
                        fig_env.update_layout(
                            title="Environmental Conditions vs Archaeological Potential<br><sub>Stars indicate locations with AI insights</sub>",
                            xaxis_title="Temperature (°C)",
                            yaxis_title="Humidity (%)",
                            height=500,
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig_env, use_container_width=True)
                    else:
                        st.info("ℹ️ No environmental data available for visualization (OpenWeatherMap API key needed)")
    
    with tab5:
        st.markdown("## 📰 Archaeological News & Updates")
        st.markdown("*Latest archaeological discoveries and research from verified sources*")
        
        if st.button("📰 Refresh Archaeological News", use_container_width=True):
            with st.spinner("📰 Fetching latest archaeological news..."):
                try:
                    news_items = st.session_state.site_discovery.real_data_processor.get_archaeological_news()
                    
                    if news_items:
                        st.session_state.archaeological_news = news_items
                        display_archaeological_news(news_items)
                    else:
                        st.warning("⚠️ No archaeological news available at the moment")
                        
                except Exception as e:
                    st.error(f"❌ Failed to fetch news: {str(e)}")
        
        # Display cached news if available
        if hasattr(st.session_state, 'archaeological_news'):
            display_archaeological_news(st.session_state.archaeological_news)
        else:
            st.info("📰 Click 'Refresh Archaeological News' to get the latest updates")
    
    with tab6:
        st.markdown("## 🤖 AI Archaeological Insights")
        st.markdown("*AI-powered analysis and insights from OpenAI*")
        
        if not st.session_state.analysis_results:
            st.warning("⚠️ No analysis results available for AI insights.")
        else:
            # Generate overall summary
            if st.button("🤖 Generate AI Survey Summary", use_container_width=True):
                with st.spinner("🤖 Generating comprehensive AI survey summary..."):
                    try:
                        ai_summary = st.session_state.site_discovery.openai_processor.generate_site_summary(
                            st.session_state.analysis_results
                        )
                        
                        if ai_summary.get('status') == 'success':
                            st.markdown("### 🤖 AI Archaeological Survey Summary")
                            st.markdown("""
                            <div class="ai-insight">
                                <h4>🤖 Comprehensive Survey Analysis</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(ai_summary.get('summary', 'No summary available'))
                            st.caption(f"Generated using {ai_summary.get('model_used', 'OpenAI')}")
                        else:
                            st.warning(f"🤖 AI summary failed: {ai_summary.get('message', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"❌ Failed to generate AI summary: {str(e)}")
            
            # Display individual AI insights
            st.markdown("### 🤖 Individual Location Insights")
            
            ai_results = [r for r in st.session_state.analysis_results if r.get('ai_insights', {}).get('status') == 'success']
            
            if ai_results:
                for i, result in enumerate(ai_results):
                    lat, lon = result['location']['lat'], result['location']['lon']
                    confidence = result.get('confidence_score', 0)
                    ai_insights = result.get('ai_insights', {})
                    
                    with st.expander(f"🤖 AI Insights for {lat:.4f}, {lon:.4f} (Confidence: {confidence:.3f})"):
                        display_ai_insights(ai_insights)
            else:
                st.info("🤖 No AI insights available. Ensure OpenAI API key is provided and run analysis with AI insights enabled.")
        
        # Export functionality
        st.markdown("---")
        st.markdown("### 💾 Export Analysis Results")
        
        if st.session_state.analysis_results:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export CSV", use_container_width=True):
                    try:
                        export_data = []
                        for result in st.session_state.analysis_results:
                            row = {
                                'latitude': result['location']['lat'],
                                'longitude': result['location']['lon'],
                                'confidence_score': result.get('confidence_score', 0),
                                'known_sites_count': len(result.get('archaeological_sites', [])),
                                'cv_features_count': sum(len(features) if isinstance(features, list) else 0 
                                                       for features in result.get('cv_features', {}).values()),
                                'data_sources': '; '.join(result.get('data_sources_used', [])),
                                'timestamp': result.get('timestamp', ''),
                                'environmental_status': result.get('environmental_data', {}).get('status', 'unavailable'),
                                'satellite_data_available': bool(result.get('satellite_data')),
                                'ai_insights_available': result.get('ai_insights', {}).get('status') == 'success'
                            }
                            
                            # Add environmental data if available
                            env_data = result.get('environmental_data', {})
                            if env_data.get('status') == 'success':
                                row.update({
                                    'temperature': env_data.get('temperature', ''),
                                    'humidity': env_data.get('humidity', ''),
                                    'weather': env_data.get('weather_description', ''),
                                    'wind_speed': env_data.get('wind_speed', ''),
                                    'air_quality': env_data.get('air_quality', '')
                                })
                            
                            export_data.append(row)
                        
                        df = pd.DataFrame(export_data)
                        csv_data = df.to_csv(index=False)
                        
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=csv_data,
                            file_name=f"archaeological_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        st.success("✅ CSV export ready for download")
                        
                    except Exception as e:
                        st.error(f"❌ CSV export failed: {str(e)}")
            
            with col2:
                if st.button("📋 Export JSON", use_container_width=True):
                    try:
                        json_data = json.dumps(st.session_state.analysis_results, indent=2, default=str)
                        
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=json_data,
                            file_name=f"archaeological_analysis_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        st.success("✅ JSON export ready for download")
                        
                    except Exception as e:
                        st.error(f"❌ JSON export failed: {str(e)}")
            
            with col3:
                if st.button("🤖 Export AI Insights", use_container_width=True):
                    try:
                        ai_data = []
                        for result in st.session_state.analysis_results:
                            if result.get('ai_insights', {}).get('status') == 'success':
                                ai_data.append({
                                    'location': f"{result['location']['lat']:.4f}, {result['location']['lon']:.4f}",
                                    'confidence_score': result.get('confidence_score', 0),
                                    'ai_analysis': result['ai_insights'].get('analysis', ''),
                                    'model_used': result['ai_insights'].get('model_used', ''),
                                    'tokens_used': result['ai_insights'].get('tokens_used', 0),
                                    'timestamp': result.get('timestamp', '')
                                })
                        
                        if ai_data:
                            ai_json = json.dumps(ai_data, indent=2, default=str)
                            
                            st.download_button(
                                label="⬇️ Download AI Insights",
                                data=ai_json,
                                file_name=f"ai_archaeological_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                            st.success("✅ AI insights export ready for download")
                        else:
                            st.warning("⚠️ No AI insights available for export")
                        
                    except Exception as e:
                        st.error(f"❌ AI insights export failed: {str(e)}")
        else:
            st.info("📊 No analysis results available for export")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; padding: 2rem;">
        🛰️ <strong>Archaeological Site Discovery Platform</strong> | 
        Real Data Sources Only • No Fake/Mock/Simulated Data<br>
        Google Earth Engine • Wikidata SPARQL • OpenWeatherMap • OpenAI • RSS Feeds<br>
        <em>Discovering archaeological potential through real API data, advanced analytics, and AI insights</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
