const chatInput = document.querySelector('.user-input');
const chatDisplay = document.querySelector('.display-area');
const inputBtn = document.querySelector('.send-button');

let userMessage;

const genChatDiv = (message, className) => {
    const chatDiv = document.createElement("div");
    chatDiv.classList.add(className);

    let chatContent = `<p>${message}</p>`;
    
    chatDiv.innerHTML = chatContent;
    return chatDiv;
}

let firstClick = true;
const handleQuery = async() => {
    userMessage = chatInput.value.trim();

    if (!userMessage) {
        return;
    }

    chatDisplay.appendChild(genChatDiv(`User: ${userMessage}`, "user-msg"));
    chatDisplay.scrollTo(0, chatDisplay.scrollHeight);

    chatInput.value = "";

    const retrievingDiv = genChatDiv(`AI: Retrieving a recent ArXiv paper related to ${userMessage}...`, "ai-msg");
    chatDisplay.appendChild(retrievingDiv);

    // Send message to FastAPI
    const response = await fetch("/retrieval", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body:   JSON.stringify({ msg : userMessage, prevChat : " "})
    }); 

    if (!response.ok) {
        throw new Error(`Response status: ${response.status}`);
    }
    
    if (response) {
        const data = await response.json();
        retrievingDiv.querySelector('p').textContent = `AI: Retrieved the paper ${data.msg}. What is your question regarding the paper?`
        firstClick = false;
    }
}

const handleQuestion = async() => {
    userMessage = chatInput.value.trim();

    if (!userMessage) {
        return;
    }

    chatDisplay.appendChild(genChatDiv(`User: ${userMessage}`, "user-msg"));
    chatDisplay.scrollTo(0, chatDisplay.scrollHeight);
    chatInput.value = "";
    const chatHistory = chatDisplay.innerText;

    const thinkingDiv = genChatDiv("Thinking...", "ai-msg");
    chatDisplay.appendChild(thinkingDiv);

    // Send message to FastAPI
    const response = await fetch("/questioning", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body:   JSON.stringify({ msg : userMessage, prevChat : chatHistory})
    }); 
    

    if (!response.ok) {
        throw new Error(`Response status: ${response.status}`);
    }

    if (response) {
        const data = await response.json();
        thinkingDiv.querySelector('p').textContent = `AI: ${data.msg}`;
    }
}

const handleClick = async() => {
    if (firstClick) {
        await handleQuery();
    } else {
        await handleQuestion();
    }
}

inputBtn.addEventListener("click", handleClick)