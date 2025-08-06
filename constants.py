import numpy as np

FILE_NAME = "Data/LungCancer_Dataset.csv"
OR_FILE_NAME= "Data/OddsRatio_Data.csv"
ALL_FEATURES = ["EPI", "AST", "AQI", "HDI", "GDP", "PWD_A", "OADR", "SR", "OOP", "PDPC", "CO2", "HUM"]
CLEANED_FEATURES=['AST', 'HDI', 'OADR', 'SR', 'OOP']
TARGET = "LCR_OR"

BETA = np.array([[13.38678921725653], [-1.7909773656139059], [1.951746012410291], [2.837287747172861], [2.8995312299033595], [-0.834747296200608]])

MEANS = np.array([[18.78957407407407, 0.760240740740741, 16.593895533333335, 19.583333333333332, 37.61342059664815]])
STDS = np.array([[6.843772055541939, 0.12913228003717525, 11.336119367246889, 10.279334716907618, 16.63743731775029]])

countries_list = [
    "Albania", "Algeria", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
    "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Benin", "Bolivia",
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Bulgaria", "Burkina Faso", "Burundi",
    "Cambodia", "Cameroon", "Canada", "Chad", "Chile", "China", "Colombia", "Comoros",
    "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark",
    "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Estonia", "Ethiopia",
    "Finland", "France", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Guatemala",
    "Guinea-Bissau", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia",
    "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan",
    "Kazakhstan", "Kenya", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho",
    "Liberia", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Mali",
    "Malta", "Mauritania", "Mauritius", "Mexico", "Moldova", "Mongolia", "Montenegro",
    "Morocco", "Myanmar", "Namibia", "Nepal", "Netherlands", "New Zealand", "Nigeria",
    "Norway", "Oman", "Pakistan", "Panama", "Paraguay", "Peru", "Philippines", "Poland",
    "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saudi Arabia", "Senegal",
    "Serbia", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "South Africa",
    "South Korea", "Spain", "Sri Lanka", "Sweden", "Switzerland", "Tanzania", "Thailand",
    "Togo", "Tunisia", "Turkey", "Turkmenistan", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan",
    "Vietnam", "Zambia", "Zimbabwe"
]


policy_recommendations = {
    ("HDI", "Too low"): [
        "Invest in public services like education, sanitation, and health.",
        "Implement social protection schemes for vulnerable populations.",
        "Use health outcomes as a central metric for development planning."
    ],
    "SR": [
        "Enforce plain packaging and restrict tobacco advertising.",
        "Implement high tobacco taxation and smoking cessation programs.",
        "Launch school-based anti-smoking campaigns."
    ],
    "OADR": [
        "Expand geriatric health and caregiving infrastructure.",
        "Encourage active aging programs to reduce dependency.",
        "Train more healthcare workers for elderly care sectors."
    ],
    "AST": [
        "Promote urban greening and reflective roofing to cool cities.",
        "Implement early warning systems for heatwaves.",
        "Adapt building codes for temperature resilience."
    ],
    
    "OOP": [
        "Validate data for potential underreporting or misclassification.",
        "Ensure government healthcare spending is effectively reaching people.",
        "Monitor for informal payments in public healthcare."
    ]
}
