class Chatbox {
  constructor() {
    this.currentMessageChunks = []; // Initialize an array to hold message chunks
      this.updateTimer = null; // Timer for updating the UI
      this.currentMessageId = null; // ID of the current message

    this.args = {
      openButton: document.querySelector(".chatbox__button"),
      chatBox: document.querySelector(".chatbox__support"),
      sendButton: document.querySelector(".send__button"),
    };

    this.state = false;
    this.messages = [];
  }

  display() {
    const { openButton, chatBox, sendButton } = this.args;

    openButton.addEventListener("click", () => this.toggleState(chatBox));

    sendButton.addEventListener("click", () => this.onSendButton(chatBox));

    const node = chatBox.querySelector("input");
    node.addEventListener("keyup", ({ key }) => {
      if (key === "Enter") {
        this.onSendButton(chatBox);
      }
    });
  }

  toggleState(chatbox) {
    this.state = !this.state;

    // show or hides the box
    if (this.state) {
      chatbox.classList.add("chatbox--active");
    } else {
      chatbox.classList.remove("chatbox--active");
    }
  }

  displayLoading(chatbox) {
    const { sendButton } = this.args;
    const loader = chatbox.querySelector("#preparingLoader");
    loader.classList.add("display_loading");
    sendButton.disabled = true;
    setTimeout(() => {
      sendButton.disabled = false;
    }, 5000);
  }

  hideLoading(chatbox) {
    const { sendButton } = this.args;
    const loader = chatbox.querySelector("#preparingLoader");
    loader.classList.remove("display_loading");
    sendButton.disabled = false;
  }
  showTypingLoading(chatbox) {
    $("#typingLoader").css("display", "block");
  }
  hideTypingLoader(chatbox) {
    $("#typingLoader").css("display", "none");
  }
  create_UUID() {
    var dt = new Date().getTime();
    var uuid = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(
      /[xy]/g,
      function (c) {
        var r = (dt + Math.random() * 16) % 16 | 0;
        dt = Math.floor(dt / 16);
        return (c == "x" ? r : (r & 0x3) | 0x8).toString(16);
      }
    );
    return uuid;
  }
  scrollToBottom(chatbox) {
    const chatmessage = chatbox.querySelector(".chatbox__messages");
    chatmessage.scrollTo(0, chatmessage.scrollHeight);
  }
  onSendButton(chatbox) {
    var textField = chatbox.querySelector("input");
    let text1 = textField.value;
    const text = '"' + text1 + '"';
    if (text === "") {
        return;
    }
    let msg1 = { name: "User", message: text1 };
    this.messages.push(msg1);
    this.updateChatText(chatbox);
    this.scrollToBottom(chatbox);
    textField.value = "";
    this.displayLoading(chatbox);

    // Close any existing EventSource
    if (this.eventSource) {
        this.eventSource.close();
    }

    // Clear the existing update timer if any
    if (this.updateTimer) {
        clearTimeout(this.updateTimer);
        this.updateTimer = null;
    }

    this.currentMessageChunks = []; // Reset the message chunks
    this.currentMessageId = this.create_UUID(); // Create a new message ID

    // Create a new EventSource
    this.eventSource = new EventSource(`http://127.0.0.1:8000/chat?input=${encodeURIComponent(text1)}`);

    this.eventSource.onmessage = (event) => {
        this.currentMessageChunks.push(event.data.replace(/<space>/g, ' '));

        if (!this.updateTimer) {
          this.updateTimer = setTimeout(() => {
              let completeChunk = this.currentMessageChunks.join('');
              this.handleCompleteChunk(completeChunk, chatbox);
              this.currentMessageChunks = [];
              this.updateTimer = null;
          }, 0); // Update UI every custom seconds
      }
      
    };

    this.eventSource.onerror = (error) => {
        console.error("EventSource failed:", error);
        this.eventSource.close();
        this.hideLoading(chatbox);
        if (this.updateTimer) {
            clearTimeout(this.updateTimer);
            this.updateTimer = null;
        }

        // Handle any remaining chunks in case of an error
        if (this.currentMessageChunks.length > 0) {
            let completeChunk = this.currentMessageChunks.join('');
            this.handleCompleteChunk(completeChunk, chatbox);
            this.currentMessageChunks = [];
        }
        this.currentMessageId = null;
    };
}

handleCompleteChunk(chunk, chatbox) {
    let msg = this.messages.find(m => m.id === this.currentMessageId);
    if (!msg) {
        msg = { id: this.currentMessageId, name: "Sam", message: chunk };
        this.messages.push(msg);
    } else {
        msg.message += chunk;
    }

    this.updateChatText(chatbox);
    this.scrollToBottom(chatbox);
}

  
  updateChatText(chatbox) {
    var html = "<p id='preparingLoader' class='loader'>Cypress is typing...</p>";
    this.messages
      .slice()
      .reverse()
      .forEach(function (item) {
        if (item.name === "Sam") {
          html += `<div class="messages__item messages__item--visitor">
            <img src="./images/cypress_logo.jpg" alt="image">
            <div style="white-space: pre-line; max-width: 100%; overflow-x: auto; word-wrap: break-word;">${item.message}</div> 
            </div>`;
        } else {
          html +=
            '<div class="messages__item messages__item--operator">' +
            '<img src="./images/avatarblue.png" alt="image">' +
            "<p>" +
            item.message +
            "</p>" +
            '<button type="button" class="edit-query">' +
            // SVG and other elements
            "</button>" +
            "</div>";
        }
      });
  
    const chatmessage = chatbox.querySelector(".chatbox__messages");
    chatmessage.innerHTML = html;
  }
  
}

const chatbox = new Chatbox();
chatbox.display();

$(document).ready(function () {
  $(".expand-button").click(function () {
    $(".chatbox").addClass("expand-chat");
    $(".expand-button").hide();
    $(".remove-expand").show();
  });
  $(".remove-expand").click(function () {
    $(".chatbox").removeClass("expand-chat");
    $(".expand-button").show();
    $(".remove-expand").hide();
  });
});
