css = '''
<style>
  .chat-message {
      padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
  }
  .chat-message.user {
      background-color: #2b313e
  }
  .chat-message.bot {
      background-color: #475063
  }
  .chat-message .avatar {
    width: 20%;
  }
  .chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
  }
  .chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
  }

div.stButton > button:first-child {
  background-color: #FF5733; /* Warm, inviting background */
  color: white; /* High contrast text */
  font-size: 32px; /* Increased font size */
  font-weight: bold; /* Bold text for emphasis */
  padding: 10px 20px; /* Larger padding for balance */
  border: none; /* Remove default border */
  border-radius: 12px; /* Smooth rounded corners */
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Add a subtle shadow for depth */
  cursor: pointer; /* Pointer cursor for interactivity */
  transition: all 0.3s ease; /* Smooth transitions for hover effects */
}

div.stButton > button:first-child:hover {
  background-color: #D2603E; /* Darken the background on hover */
  box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2); /* Enhance shadow on hover */
  transform: scale(1.1); /* Slightly enlarge on hover */
}

 
</style>
'''