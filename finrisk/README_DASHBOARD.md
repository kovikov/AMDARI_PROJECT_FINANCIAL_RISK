# FinRisk Streamlit Dashboard

A comprehensive real-time monitoring dashboard for the FinRisk application, providing insights into credit risk, fraud detection, and portfolio analysis.

## 🚀 Features

### 📊 Overview Dashboard
- **Key Metrics**: Total customers, average credit score, approval rate, fraud rate
- **Trend Analysis**: Credit applications and transaction volume over time
- **Risk Distribution**: Customer segmentation by risk level

### 💳 Credit Risk Analysis
- **Application Metrics**: Total applications, approval/rejection rates
- **Trend Analysis**: Credit application trends with approval rates
- **Risk Segmentation**: Distribution and analysis by risk segments

### 🚨 Fraud Detection Analysis
- **Transaction Monitoring**: Total transactions, fraud detection rates
- **Loss Analysis**: Transaction volume vs fraud losses
- **Trend Analysis**: Daily fraud rates and patterns

### 📈 Portfolio Analysis
- **Portfolio Overview**: Total customers, portfolio value, average credit score
- **Risk Analysis**: Distribution by risk segments with value analysis
- **Detailed Metrics**: Comprehensive risk segment breakdown

### 🤖 Model Performance Analysis
- **Model Metrics**: Total predictions, average confidence, active models
- **Performance Trends**: Model-specific performance over time
- **Model Comparison**: Side-by-side model performance analysis

### ⚙️ System Health
- **Database Status**: Real-time database connectivity monitoring
- **Cache Statistics**: Redis cache performance metrics
- **System Metrics**: Overall system health indicators

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PostgreSQL database with FinRisk schema
- Streamlit and other dependencies

### Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements_dashboard.txt
   ```

2. **Database Configuration**:
   Ensure your PostgreSQL database is running and accessible with the following credentials:
   - Database: `amdari_project`
   - User: `postgres`
   - Password: `Kovikov1978@`
   - Host: `localhost`
   - Port: `5432`

3. **Data Loading**:
   Make sure you have loaded the seed data using the enhanced data loader:
   ```bash
   python enhanced_data_loader.py
   ```

## 🚀 Running the Dashboard

### Start the Dashboard
```bash
streamlit run dashboard.py
```

The dashboard will be available at `http://localhost:8501`

### Command Line Options
- **Auto-refresh**: Enable automatic data refresh every 30 seconds
- **Date Range**: Select custom date ranges for analysis
- **Navigation**: Switch between different dashboard sections

## 📊 Dashboard Sections

### 1. Overview Dashboard
- **Real-time Metrics**: Key performance indicators at a glance
- **Trend Charts**: Visual representation of credit and fraud trends
- **Risk Distribution**: Pie chart showing customer risk segmentation

### 2. Credit Risk Analysis
- **Application Trends**: Daily credit application patterns
- **Approval Rates**: Time-series analysis of approval rates
- **Risk Segmentation**: Bar charts showing customer distribution by risk

### 3. Fraud Detection Analysis
- **Transaction Monitoring**: Real-time transaction volume and fraud detection
- **Loss Analysis**: Financial impact of fraud detection
- **Trend Analysis**: Fraud rate patterns over time

### 4. Portfolio Analysis
- **Portfolio Overview**: High-level portfolio metrics
- **Risk Analysis**: Detailed risk segment analysis with value distribution
- **Customer Insights**: Comprehensive customer risk profiling

### 5. Model Performance Analysis
- **Model Metrics**: Performance indicators for all active models
- **Trend Analysis**: Model performance over time
- **Comparison**: Side-by-side model performance comparison

### 6. System Health
- **Database Status**: Real-time connectivity monitoring
- **Cache Performance**: Redis cache statistics and performance
- **System Metrics**: Overall system health and performance

## 🔧 Configuration

### Database Connection
The dashboard uses direct PostgreSQL connections with the following configuration:
```python
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "amdari_project",
    "user": "postgres",
    "password": "Kovikov1978@"
}
```

### Caching
- **Data Caching**: 5-10 minute cache for database queries
- **Resource Caching**: Persistent caching for application settings
- **Auto-refresh**: Optional 30-second auto-refresh for real-time updates

## 📈 Data Sources

### Portfolio Metrics
- **Source**: `customer_profiles` table
- **Metrics**: Customer count, credit scores, income analysis
- **Refresh**: 5-minute cache

### Credit Metrics
- **Source**: `credit_applications` table
- **Metrics**: Application trends, approval rates
- **Refresh**: 5-minute cache

### Fraud Metrics
- **Source**: `transaction_data` table
- **Metrics**: Transaction volume, fraud detection
- **Refresh**: 5-minute cache

### Risk Distribution
- **Source**: `customer_profiles` table
- **Metrics**: Risk segmentation analysis
- **Refresh**: 10-minute cache

### Model Predictions
- **Source**: `model_predictions` table
- **Metrics**: Model performance and predictions
- **Refresh**: 5-minute cache

## 🧪 Testing

### Run Dashboard Tests
```bash
python test_dashboard.py
```

### Test Coverage
- ✅ Database connection
- ✅ Dashboard imports
- ✅ Portfolio metrics loading
- ✅ Credit metrics loading
- ✅ Fraud metrics loading
- ✅ Risk distribution loading
- ✅ Model predictions loading
- ✅ Cache functionality

## 🎨 Customization

### Styling
The dashboard uses custom CSS for enhanced visual appeal:
- **Metric Cards**: Styled containers for key metrics
- **Alert Classes**: Color-coded alerts for different risk levels
- **Responsive Layout**: Wide layout optimized for monitoring

### Charts
- **Plotly Integration**: Interactive charts with zoom and hover capabilities
- **Subplot Support**: Multi-axis charts for trend analysis
- **Color Schemes**: Consistent color coding across all visualizations

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Failed**:
   - Verify PostgreSQL is running
   - Check database credentials
   - Ensure database `amdari_project` exists

2. **No Data Displayed**:
   - Run the enhanced data loader first
   - Check if tables contain data
   - Verify date ranges in queries

3. **Cache Issues**:
   - Redis connection is optional (uses mock client)
   - Dashboard works without Redis
   - Check cache configuration if needed

4. **Import Errors**:
   - Install all required dependencies
   - Check Python version compatibility
   - Verify Streamlit installation

### Performance Optimization
- **Query Optimization**: All queries are optimized for performance
- **Caching Strategy**: Intelligent caching reduces database load
- **Lazy Loading**: Data loaded only when needed

## 📝 API Integration

The dashboard can be extended to integrate with:
- **FastAPI Endpoints**: Real-time data from API services
- **External Data Sources**: Additional data feeds
- **Alert Systems**: Integration with monitoring systems

## 🔮 Future Enhancements

### Planned Features
- **Real-time Alerts**: Push notifications for critical events
- **Export Functionality**: PDF/Excel report generation
- **User Authentication**: Multi-user support with roles
- **Advanced Analytics**: Machine learning insights
- **Mobile Support**: Responsive design for mobile devices

### Performance Improvements
- **Query Optimization**: Further database query optimization
- **Caching Enhancement**: Advanced caching strategies
- **Load Balancing**: Support for high-traffic scenarios

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Run the test suite to verify functionality
3. Review database connectivity and data availability
4. Check Streamlit and dependency versions

## 📄 License

This dashboard is part of the FinRisk application suite and follows the same licensing terms.
