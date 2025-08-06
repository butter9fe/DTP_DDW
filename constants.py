ALL_FEATURES = ["HDI", "SR", "EPI", "AQI", "PWD_A", "OADR", "GDP", "AST", "OOP"]
TARGET = "LCR_OR"
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
    ("HDI", "Too high"): [
        "Ensure high development translates into equitable health outcomes.",
        "Focus on marginalized communities that may be left behind.",
        "Encourage sustainability practices in high-HDI regions."
    ],
    ("HDI", "Too low"): [
        "Invest in public services like education, sanitation, and health.",
        "Implement social protection schemes for vulnerable populations.",
        "Use health outcomes as a central metric for development planning."
    ],
    ("SR", "Too high"): [
        "Enforce plain packaging and restrict tobacco advertising.",
        "Implement high tobacco taxation and smoking cessation programs.",
        "Launch school-based anti-smoking campaigns."
    ],
    ("SR", "Too low"): [
        "Investigate under-reporting or cultural stigma in surveys.",
        "Monitor e-cigarette or alternative tobacco product use.",
        "Ensure youth education continues to promote non-smoking habits."
    ],
    ("EPI", "Too high"): [
        "Maintain strong environmental protections with robust enforcement.",
        "Share best practices across sectors and neighboring countries.",
        "Monitor for performance consistency across regions."
    ],
    ("EPI", "Too low"): [
        "Improve regulatory frameworks for environmental protection.",
        "Invest in air, water, and land quality monitoring systems.",
        "Enforce environmental impact assessments on new developments."
    ],
    ("AQI", "Too high"): [
        "Introduce vehicle emissions inspections and low-emission zones.",
        "Shift urban transport to electric and green alternatives.",
        "Ban open-air waste burning and enforce clean fuel use."
    ],
    ("AQI", "Too low"): [
        "Validate AQI values with independent monitoring systems.",
        "Investigate if data under-reports peak pollution periods.",
        "Maintain existing air quality efforts while planning urban expansion."
    ],
    ("PWD_A", "Too high"): [
        "Encourage suburban development to ease core urban congestion.",
        "Invest in mass transit to prevent crowding-related health risks.",
        "Promote green and open public spaces in dense cities."
    ],
    ("PWD_A", "Too low"): [
        "Revitalize underpopulated areas through incentives for businesses.",
        "Support balanced population distribution through rural development.",
        "Invest in telemedicine and mobile clinics in sparse regions."
    ],
    ("OADR", "Too high"): [
        "Expand geriatric health and caregiving infrastructure.",
        "Encourage active aging programs to reduce dependency.",
        "Train more healthcare workers for elderly care sectors."
    ],
    ("OADR", "Too low"): [
        "Ensure youth support systems consider future aging trends.",
        "Balance investment across age groups in public policy.",
        "Promote intergenerational integration in housing and services."
    ],
    ("GDP", "Too high"): [
        "Ensure economic growth translates to improved health equity.",
        "Avoid neglecting environmental standards for GDP growth.",
        "Reinvest gains into sustainable health and green infrastructure."
    ],
    ("GDP", "Too low"): [
        "Support inclusive economic policies with health co-benefits.",
        "Invest in job creation in green and care industries.",
        "Link development funding to public health outcomes."
    ],
    ("AST", "Too high"): [
        "Promote urban greening and reflective roofing to cool cities.",
        "Implement early warning systems for heatwaves.",
        "Adapt building codes for temperature resilience."
    ],
    ("AST", "Too low"): [
        "Ensure buildings meet thermal comfort standards.",
        "Promote insulation and heating access in colder zones.",
        "Support vulnerable groups during cold seasons with aid."
    ],
    ("OOP", "Too high"): [
        "Expand universal health coverage and reduce co-payments.",
        "Subsidize essential medicine and chronic disease treatment.",
        "Strengthen public healthcare systems to lower private costs."
    ],
    ("OOP", "Too low"): [
        "Validate data for potential underreporting or misclassification.",
        "Ensure government healthcare spending is effectively reaching people.",
        "Monitor for informal payments in public healthcare."
    ]
}
