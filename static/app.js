async function fetchDrafts() {
    const res = await fetch("/pending-drafts");
    const drafts = await res.json();
    const container = document.getElementById("drafts-container");
    container.innerHTML = "";

    Object.entries(drafts).forEach(([id, draft]) => {
        const div = document.createElement("div");
        div.classList.add("draft");

        div.innerHTML = `
            <h3>${draft.subject || "No Subject"}</h3>
            <p><b>Original Email:</b> ${draft.body}</p>
            <textarea id="reply-${id}">${draft.ai_draft}</textarea>
            <br>
            <button onclick="approveDraft('${id}')">Approve</button>
            <button onclick="rejectDraft('${id}')">Reject</button>
        `;
        container.appendChild(div);
    });
}

async function approveDraft(id) {
    const reply = document.getElementById(`reply-${id}`).value;
    const res = await fetch("/save-reply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email_id: id, your_reply: reply })
    });
    if (res.ok) {
        alert("Reply approved and saved!");
        fetchDrafts();
    } else {
        alert("Error approving draft");
    }
}

function rejectDraft(id) {
    alert(`Draft ${id} rejected (not implemented yet).`);
    // Optional: Add API to remove from pending_drafts without saving
}

fetchDrafts();

