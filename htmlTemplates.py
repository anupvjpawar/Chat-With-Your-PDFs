css = """
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f5f5;
}

.stButton>button {
    color: white;
    background-color: #007bff;
    border: none;
    padding: 10px 24px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 8px;
}

h4 {
    margin-bottom: 15px;
}

.chat-bubble {
    display: inline-block;
    max-width: 70%;
    padding: 10px 20px;
    border-radius: 20px;
    margin-bottom: 10px;
}

.user-bubble {
    background-color: #007bff;
    color: white;
    margin-left: auto;
}

.bot-bubble {
    background-color: #e1e1e1;
    color: black;
}

</style>
"""

bot_template = """
<div class="chat-bubble bot-bubble">
    {{MSG}}
</div>
"""

user_template = """
<div class="chat-bubble user-bubble">
    {{MSG}}
</div>
"""
