// =====================
// CONFIG
// =====================
const BACKEND_URL = "http://127.0.0.1:8000";

// =====================
// Helpers
// =====================
function showFlash(message, type = "success") {
  const box = document.getElementById("flash");
  box.textContent = message;
  box.classList.remove("hidden", "success", "error");
  box.classList.add(type);
  setTimeout(() => {
    box.classList.add("hidden");
  }, 4000);
}

function formatDateTime(dtString) {
  const d = new Date(dtString);
  if (isNaN(d)) return dtString;
  return d.toLocaleString();
}

// =====================
// Dashboard Stats
// =====================
async function loadSummaryStats() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/stats/summary`);
    const data = await res.json();
    document.getElementById("total-scans").textContent =
      data.total_scans ?? data.total ?? "0";
    document.getElementById("delivered-count").textContent =
      data.delivered ?? "0";
    document.getElementById("pending-count").textContent =
      data.pending ?? "0";
    document.getElementById("alert-count").textContent =
      data.alerts ?? "0";
  } catch (err) {
    console.error(err);
    showFlash("Failed to load stats", "error");
  }
}

// =====================
// Recent Transactions
// =====================
async function loadRecentTransactions() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/transactions/recent?limit=10`);
    const data = await res.json();
    const tbody = document.getElementById("tx-table-body");
    tbody.innerHTML = "";

    data.forEach((tx) => {
      const tr = document.createElement("tr");

      const status = tx.status || "pending";
      const statusClass =
        status === "delivered"
          ? "status-delivered"
          : status === "pending"
          ? "status-pending"
          : "status-alert";

      tr.innerHTML = `
        <td>${tx.id}</td>
        <td>${tx.item_id}</td>
        <td>${tx.dealer_code}</td>
        <td>${tx.farmer_code}</td>
        <td>
          <span class="status-badge ${statusClass}">
            ${status.charAt(0).toUpperCase() + status.slice(1)}
          </span>
        </td>
        <td>${formatDateTime(tx.timestamp)}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error(err);
    showFlash("Failed to load recent transactions", "error");
  }
}

// =====================
// Record Transaction (Dealer form)
// =====================
function setupTransactionForm() {
  const form = document.getElementById("tx-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const dealerCode = document.getElementById("dealer-code").value.trim();
    const farmerCode = document.getElementById("farmer-code").value.trim();
    const itemId = document.getElementById("item-id").value.trim();

    if (!dealerCode || !farmerCode || !itemId) return;

    try {
      const res = await fetch(`${BACKEND_URL}/api/transactions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dealer_code: dealerCode,
          farmer_code: farmerCode,
          item_id: itemId,
          status: "pending",
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Failed to record");
      }

      showFlash("Transaction recorded ✅", "success");
      form.reset();
      loadSummaryStats();
      loadRecentTransactions();
    } catch (err) {
      console.error(err);
      showFlash(err.message || "Error recording transaction", "error");
    }
  });
}

// =====================
// Farmers in radius
// =====================
function setupRadiusForm() {
  const form = document.getElementById("radius-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const dealerCode = document
      .getElementById("radius-dealer-code")
      .value.trim();
    const radiusKm = Number(document.getElementById("radius-km").value || 10);
    if (!dealerCode) return;

    try {
      const res = await fetch(
        `${BACKEND_URL}/api/dealers/${encodeURIComponent(
          dealerCode
        )}/farmer-stats?radius_km=${radiusKm}`
      );
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Failed to fetch radius stats");
      }
      const data = await res.json();

      document.getElementById("farmers-24h").textContent =
        data.unique_farmers_last_24h ?? data.uniqueFarmersLast24h ?? "0";
      document.getElementById("farmers-7d").textContent =
        data.unique_farmers_last_7d ?? data.uniqueFarmersLast7d ?? "0";
      document.getElementById("avg-per-day").textContent =
        (data.avg_per_day ?? data.avgPerDay ?? 0).toFixed(2);

      const statusEl = document.getElementById("spike-status");
      let text = "Normal pattern ✅";
      let color = "#166534";

      if (data.is_spike ?? data.isSpike) {
        text = "Spike in sales 🚨";
        color = "#b91c1c";
      } else if (data.is_low ?? data.isLow) {
        text = "Unusually low sales ⚠️";
        color = "#92400e";
      }

      statusEl.textContent = text;
      statusEl.style.color = color;
    } catch (err) {
      console.error(err);
      showFlash(err.message || "Error loading radius stats", "error");
    }
  });
}

// =====================
// ML Anomalies
// =====================
async function loadMLAnomalies(dealerCode) {
  const list = document.getElementById("ml-alert-list");
  list.innerHTML = "";

  if (!dealerCode) {
    list.innerHTML = "<li class='alert-item'>Enter a dealer code to see ML alerts.</li>";
    return;
  }

  try {
    const res = await fetch(
      `${BACKEND_URL}/api/ml/dealers/${encodeURIComponent(
        dealerCode
      )}/daily-anomalies`
    );
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Failed to fetch ML anomalies");
    }
    const data = await res.json();

    if (!data.days || data.days.length === 0) {
      list.innerHTML = "<li class='alert-item'>No data / anomalies for this dealer.</li>";
      return;
    }

    data.days.forEach((day) => {
      const li = document.createElement("li");
      li.className = "alert-item";
      const isSuspicious = day.status === "suspicious";
      li.innerHTML = `
        <strong>${day.date}</strong>
        <div>Unique farmers: ${day.unique_farmers_24h}</div>
        <div>Total transactions: ${day.total_transactions_24h}</div>
        <small>Status: ${
          isSuspicious ? "Suspicious 🚨" : "Normal ✅"
        } | Score: ${day.anomaly_score}</small>
      `;
      list.appendChild(li);
    });
  } catch (err) {
    console.error(err);
    showFlash(err.message || "Error loading ML alerts", "error");
  }
}

function setupMLControls() {
  const btn = document.getElementById("load-ml");
  btn.addEventListener("click", () => {
    const dealerCode = document
      .getElementById("ml-dealer-code")
      .value.trim();
    loadMLAnomalies(dealerCode);
  });
}

// =====================
// Init
// =====================
function init() {
  loadSummaryStats();
  loadRecentTransactions();
  setupTransactionForm();
  setupRadiusForm();
  setupMLControls();

  document
    .getElementById("refresh-tx")
    .addEventListener("click", loadRecentTransactions);

  // Auto-fill some defaults for faster demo
  document.getElementById("radius-dealer-code").value = "D123";
  document.getElementById("ml-dealer-code").value = "D123";
}

document.addEventListener("DOMContentLoaded", init);