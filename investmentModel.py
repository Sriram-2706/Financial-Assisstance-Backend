import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy.optimize import minimize
# import google as genai
from gemini import generate

# Configure Google Gemini API (Use your API Key)
# genai.configure(api_key="YOUR_GEMINI_API_KEY")

# Market Indices to Track
MARKET_INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN"
}

# Expected Returns & Covariance for Portfolio Optimization
EXPECTED_RETURNS = np.array([0.08, 0.04, 0.12])
COVARIANCE_MATRIX = np.array([
    [0.10, 0.02, 0.04],
    [0.02, 0.05, 0.01],
    [0.04, 0.01, 0.20]
])
TARGET_RETURNS = {"low": 0.04, "medium": 0.06, "high": 0.08}

# ✅ Fetch and Plot Market Indices
def plot_market_indices():
    plt.figure(figsize=(10, 5))
    for name, ticker in MARKET_INDICES.items():
        data = yf.download(ticker, period="1y")["Close"]
        data.plot(label=name)
    plt.title("Stock Market Index Trends (Last 1 Year)")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.grid()
    
    # Convert plot to Base64 image
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    base64_img = base64.b64encode(img.read()).decode('utf-8')
    plt.close()
    
    return base64_img

# ✅ Portfolio Optimization
def optimize_portfolio(target_return, expected_returns, cov_matrix):
    n = len(expected_returns)
    init_guess = np.repeat(1/n, n)
    
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: np.dot(w, expected_returns) - target_return}
    ]
    
    bounds = tuple((0, 1) for _ in range(n))
    
    result = minimize(lambda w: np.dot(w.T, np.dot(cov_matrix, w)), init_guess,
                      method="SLSQP", bounds=bounds, constraints=constraints)
    
    return result.x if result.success else init_guess

# ✅ Simulating Portfolio Growth with Monte Carlo Simulation
def simulate_portfolio_growth(initial_investment, portfolio_weights, years=10, simulations=1000):
    np.random.seed(42)
    
    mean_returns = EXPECTED_RETURNS
    cov_matrix = COVARIANCE_MATRIX

    portfolio_returns = []

    for _ in range(simulations):
        annual_returns = np.random.multivariate_normal(mean_returns, cov_matrix, years)
        portfolio_growth = initial_investment

        for year in range(years):
            portfolio_growth *= (1 + np.dot(annual_returns[year], portfolio_weights))

        portfolio_returns.append(portfolio_growth)

    # Plot Simulation
    plt.figure(figsize=(10, 5))
    sns.histplot(portfolio_returns, kde=True, bins=50)
    plt.axvline(initial_investment, color='r', linestyle='dashed', linewidth=2)
    plt.title("Monte Carlo Simulation of Portfolio Growth")
    plt.xlabel("Final Portfolio Value (₹)")
    plt.ylabel("Frequency")
    
    # Convert plot to Base64 image
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    base64_img = base64.b64encode(img.read()).decode('utf-8')
    plt.close()

    return base64_img

# ✅ AI-Generated Investment Insights using Gemini
def get_gemini_insights(customer_data, portfolio_allocation):
    prompt = (
        "You are a seasoned financial advisor with deep expertise in personalized wealth management. "
        "Based on the following customer profile and portfolio allocation, provide clear, and concise and short actionable investment advice.\n\n"
        f"- Monthly Income: ₹{customer_data['income']:.2f}\n"
        f"- Monthly Expenses: ₹{customer_data['expenses']:.2f}\n"
        f"- Current Investments: ₹{customer_data['current_investment']:.2f}\n"
        f"- Risk Tolerance: {customer_data['risk_tolerance']}\n"
        f"- Investment Horizon: {customer_data['investment_horizon']} years\n\n"
        "Computed Portfolio Allocation:\n"
        f"- Stocks: {portfolio_allocation[0]*100:.1f}%\n"
        f"- Bonds: {portfolio_allocation[1]*100:.1f}%\n"
        f"- Alternatives: {portfolio_allocation[2]*100:.1f}%\n\n"
        "Provide recommendations that account for the customer's financial ability, potential growth, "
        "and long-term needs."
        "Your Total text response should not exceed 300 words"
    )
    
    try:
        # model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")  # Use latest available model
    #    response = model.generate_content(prompt)
    #    return response.text.strip() if response else "No response from AI"

        responseText = generate(prompt)
        return responseText if responseText else "No response from AI"
    except Exception as e:
        return f"Error in Gemini API call: {e}"

# ✅ Build Investment Plan Function
def build_investment_plan(income, expenses, current_investment, risk_tolerance, investment_horizon):
    savings = income - expenses - current_investment
    target_return = TARGET_RETURNS.get(risk_tolerance.lower(), 0.06)
    optimal_weights = optimize_portfolio(target_return, EXPECTED_RETURNS, COVARIANCE_MATRIX)

    # Get Market Trends & Simulation Results
    market_trends = plot_market_indices()
    simulation_image = simulate_portfolio_growth(savings, optimal_weights, investment_horizon)

    # Get AI Insights
    ai_insights = get_gemini_insights({
        "income": income,
        "expenses": expenses,
        "current_investment": current_investment,
        "risk_tolerance": risk_tolerance,
        "investment_horizon": investment_horizon
    }, optimal_weights)

    return {
        "portfolio_allocation": {
            "stocks": round(optimal_weights[0] * 100, 1),
            "bonds": round(optimal_weights[1] * 100, 1),
            "alternatives": round(optimal_weights[2] * 100, 1)
        },
        "market_trends": market_trends,
        "monte_carlo_simulation": simulation_image,
        "ai_insights": ai_insights
    }
