import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from datetime import datetime, timezone, timedelta

# ------------------------------------------------------------------------------
# DATABASE CONNECTION
# ------------------------------------------------------------------------------
DB_CONNECTION_STRING = (
    "postgresql://spellbound-discord_owner:ruq1GQfcRo2I"
    "@ep-calm-mountain-a5oul4yq.us-east-2.aws.neon.tech/"
    "spellbound-discord?sslmode=require"
)

@st.cache_data
def get_data():
    """Fetch joined message and analysis data from the database."""
    engine = create_engine(DB_CONNECTION_STRING)
    with engine.connect() as conn:
        query = text("""
            SELECT 
                m.id AS message_id,
                m.content,
                m.timestamp,
                u.username AS author_username,
                c.name AS channel_name,
                ma.summary,
                ma.sentiment_thought_rationale,
                ma.sentiment_score,
                ma.relevance_thought_rationale,
                ma.relevance_score,
                ma.competitor_mentions,
                ma.product_praise,
                ma.product_complaints,
                ma.product_technical_problem,
                ma.feature_requests,
                ma.product_justification
            FROM messages m
            JOIN message_analytics ma ON m.id = ma.message_id
            JOIN users u ON m.author_id = u.id
            JOIN channels c ON m.channel_id = c.id
            ORDER BY m.timestamp DESC
        """)
        df = pd.read_sql(query, conn)
    return df

# ------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------
def humanize_timedelta(delta: timedelta) -> str:
    """Convert a timedelta into a human-friendly string like '2 days 3 hours ago'."""
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:  # Future date
        return "in the future"

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60

    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days > 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")

    if not parts:
        return "just now"
    return " ".join(parts) + " ago"

def apply_filters(
    df: pd.DataFrame,
    author_filter: str,
    content_filter: str,
    channel_filter: str,
    sentiment_min: float,
    sentiment_max: float,
    relevance_min: float,
    relevance_max: float
) -> pd.DataFrame:
    """Apply a series of filters on the DataFrame in-memory."""
    # Author
    if author_filter.strip():
        df = df[df["author_username"].str.contains(author_filter, case=False, na=False)]
    # Content
    if content_filter.strip():
        df = df[df["content"].str.contains(content_filter, case=False, na=False)]
    # Channel
    if channel_filter.strip():
        df = df[df["channel_name"].str.contains(channel_filter, case=False, na=False)]
    # Scores
    df = df[df["sentiment_score"].between(sentiment_min, sentiment_max)]
    df = df[df["relevance_score"].between(relevance_min, relevance_max)]
    return df

def show_message_details(row: pd.Series):
    """Renders a detailed drilldown of a single message row in Streamlit."""
    # CONTENT
    st.markdown("### Content")
    raw_content = row["content"]
    lines = raw_content.splitlines()
    if not lines:
        lines = [raw_content]
    quoted_block = "\n".join(f"> {l}" for l in lines if l.strip()) or f"> {raw_content}"
    st.markdown(quoted_block)

    # SUMMARY
    st.markdown("#### Summary")
    st.markdown(row["summary"] if row["summary"] else "_No summary_")

    # Single line metadata
    now = datetime.now(timezone.utc)
    dt = row["timestamp"]
    delta = now - dt
    rel_time = humanize_timedelta(delta)

    meta_line = (
        f"Author: {row['author_username']} &nbsp;|&nbsp;"
        f"Channel: {row['channel_name']} &nbsp;|&nbsp;"
        f"Time: {rel_time} &nbsp;&nbsp;"
        f"Message ID: `{row['message_id']}`"
    )
    st.markdown(f"<small>{meta_line}</small>", unsafe_allow_html=True)

    # SENTIMENT
    st.markdown("### Sentiment")
    c1, c2 = st.columns([0.5, 1.5], gap="small")
    with c1:
        st.metric("Score", f"{row['sentiment_score']:.2f}")
    with c2:
        st.markdown(f"**Rationale**: {row['sentiment_thought_rationale']}")

    # RELEVANCE
    st.markdown("### Relevance")
    c3, c4 = st.columns([0.5, 1.5], gap="small")
    with c3:
        st.metric("Score", f"{row['relevance_score']:.2f}")
    with c4:
        st.markdown(f"**Rationale**: {row['relevance_thought_rationale']}")

def filter_and_sort_by_sent_rel_date(
    df: pd.DataFrame,
    sentiment_min: float,
    sentiment_max: float,
    relevance_min: float,
    relevance_max: float,
    sort_option: str
) -> pd.DataFrame:
    """Filter by sentiment/relevance and then sort by date, sentiment, or relevance."""
    # Filter
    filtered = df[
        (df["sentiment_score"].between(sentiment_min, sentiment_max)) &
        (df["relevance_score"].between(relevance_min, relevance_max))
    ]

    # Sort
    if sort_option == "Date (Newest First)":
        filtered = filtered.sort_values(by="timestamp", ascending=False)
    elif sort_option == "Date (Oldest First)":
        filtered = filtered.sort_values(by="timestamp", ascending=True)
    elif sort_option == "Sentiment (High→Low)":
        filtered = filtered.sort_values(by="sentiment_score", ascending=False)
    elif sort_option == "Sentiment (Low→High)":
        filtered = filtered.sort_values(by="sentiment_score", ascending=True)
    elif sort_option == "Relevance (High→Low)":
        filtered = filtered.sort_values(by="relevance_score", ascending=False)
    elif sort_option == "Relevance (Low→High)":
        filtered = filtered.sort_values(by="relevance_score", ascending=True)

    return filtered

def show_category_tab_as_cards(df: pd.DataFrame, column_name: str, label_for_tab: str):
    """
    Displays a tab for the given column_name where that column is expected to be an array.
    Instead of a table, each item in the array is listed as a separate "card" with:
      - The item text
      - Short excerpt of the original message
      - A small data line with author, channel, time, sentiment, relevance
    There is a filter by sentiment/relevance and a sort option.
    """
    st.subheader(f"{label_for_tab}")
    
    if df.empty:
        st.info("No data available overall.")
        return
    
    # Filter interface
    # Defaults
    sentiment_min_def = float(df["sentiment_score"].min())
    sentiment_max_def = float(df["sentiment_score"].max())
    relevance_min_def = float(df["relevance_score"].min())
    relevance_max_def = float(df["relevance_score"].max())

    colA, colB = st.columns(2)
    with colA:
        sentiment_min, sentiment_max = st.slider(
            "Sentiment Score Range",
            min_value=0.0,
            max_value=5.0,
            value=(sentiment_min_def, sentiment_max_def),
            step=0.01,
            key=f"{column_name}_sent_slider"
        )
    with colB:
        relevance_min, relevance_max = st.slider(
            "Relevance Score Range",
            min_value=0.0,
            max_value=5.0,
            value=(relevance_min_def, relevance_max_def),
            step=0.01,
            key=f"{column_name}_rel_slider"
        )
    sort_option = st.selectbox(
        "Sort by",
        [
            "Date (Newest First)",
            "Date (Oldest First)",
            "Sentiment (High→Low)",
            "Sentiment (Low→High)",
            "Relevance (High→Low)",
            "Relevance (Low→High)",
        ],
        key=f"{column_name}_sort_select"
    )
    
    # Filter and sort
    filtered = filter_and_sort_by_sent_rel_date(
        df, 
        sentiment_min, sentiment_max, 
        relevance_min, relevance_max, 
        sort_option
    )
    
    # If the user is showing e.g. Competitor Mentions, we want to list them
    # For each row, if the array is not empty, show a card for each item
    results_found = False
    for _, row in filtered.iterrows():
        items_list = row[column_name]
        # Some could be None or empty
        if not isinstance(items_list, list) or len(items_list) == 0:
            continue
        
        # For each item in that list, produce a "card"
        for item in items_list:
            results_found = True
            snippet = row["content"].replace("\n", " ")
            max_len = 500
            if len(snippet) > max_len:
                snippet = snippet[:max_len] + "..."
            
            now = datetime.now(timezone.utc)
            dt = row["timestamp"]
            delta = now - dt
            rel_time = humanize_timedelta(delta)

            st.write("-----")
            # Main item text
            st.markdown(f"### {item}")
            # Short excerpt
            st.markdown(f"**Message excerpt:** {snippet}")
            # Small data line
            st.markdown(
                f"**Author**: {row['author_username']} | "
                f"**Channel**: {row['channel_name']} | "
                f"**Time**: {rel_time} | "
                f"**Sentiment**: {row['sentiment_score']:.2f} | "
                f"**Relevance**: {row['relevance_score']:.2f}"
            )
    if not results_found:
        st.info(f"No messages have {label_for_tab} items under these filters/sort settings.")

# ------------------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Message Analysis Explorer", layout="wide")
    # We only have the main tab plus the 6 category tabs:
    tabs = st.tabs([
        "Main Analysis Explorer",
        "Competitor Mentions",
        "Product Praise",
        "Product Complaints",
        "Product Technical Problems",
        "Feature Requests",
        "Product Justification"
    ])
    
    with tabs[0]:
        st.title("Message & Analysis Explorer")
        
        df = get_data()
        if df.empty:
            st.warning("No data found.")
            return
        
        # ---------------------
        # Filtering Interface
        # ---------------------
        st.subheader("Filtering Panel")
        with st.expander("Click to show filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                author_filter = st.text_input("Filter by Author Name (contains)", value="")
                content_filter = st.text_input("Filter by Message Content (contains)", value="")
                channel_filter = st.text_input("Filter by Channel Name (contains)", value="")
            with col2:
                sentiment_min_def = float(df["sentiment_score"].min()) if not df.empty else 0.0
                sentiment_max_def = float(df["sentiment_score"].max()) if not df.empty else 5.0
                sentiment_min, sentiment_max = st.slider(
                    "Sentiment Score Range",
                    min_value=0.0,
                    max_value=5.0,
                    value=(sentiment_min_def, sentiment_max_def),
                    step=0.01
                )
            with col3:
                relevance_min_def = float(df["relevance_score"].min()) if not df.empty else 0.0
                relevance_max_def = float(df["relevance_score"].max()) if not df.empty else 5.0
                relevance_min, relevance_max = st.slider(
                    "Relevance Score Range",
                    min_value=0.0,
                    max_value=5.0,
                    value=(relevance_min_def, relevance_max_def),
                    step=0.01
                )

        # Apply top-level filters
        filtered_df = apply_filters(
            df,
            author_filter,
            content_filter,
            channel_filter,
            sentiment_min,
            sentiment_max,
            relevance_min,
            relevance_max
        )

        # OPEN 2 COLUMNS
        col_left, col_right = st.columns([1,2], gap="medium")

        with col_left:
            st.subheader("Filtered Messages")
            if filtered_df.empty:
                st.info("No messages match your filter.")
                st.session_state["selected_msg_id"] = None
            else:
                # Initialize session state
                if "selected_msg_id" not in st.session_state:
                    st.session_state["selected_msg_id"] = None
                
                # Create a truncated column for display
                table_df = filtered_df.copy()
                table_df["Truncated"] = table_df["content"].apply(
                    lambda x: x.replace("\n"," ")[:80] + ("..." if len(x) > 80 else "")
                )
                table_df["Selected"] = False

                # Restore existing selection if present
                if st.session_state["selected_msg_id"] in table_df["message_id"].values:
                    idx = table_df[table_df["message_id"] == st.session_state["selected_msg_id"]].index
                    table_df.loc[idx, "Selected"] = True

                table_df.set_index("message_id", inplace=True, drop=False)

                returned_df = st.data_editor(
                    table_df[["Selected", "Truncated"]],
                    column_config={
                        "Selected": st.column_config.CheckboxColumn(
                            "Select",
                            help="Check to select this row (single selection)."
                        ),
                        "Truncated": st.column_config.TextColumn(
                            "Message (Truncated)",
                            help="Shortened version of message content",
                            disabled=True
                        )
                    },
                    num_rows="fixed",
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                    key="main_data_editor"
                )

                # Single-row selection logic
                if not returned_df.equals(table_df[["Selected","Truncated"]]):
                    selected_indexes = returned_df.index[returned_df["Selected"] == True].tolist()
                    if len(selected_indexes) > 1:
                        last_idx = selected_indexes[-1]
                        for i in selected_indexes:
                            if i != last_idx:
                                returned_df.at[i, "Selected"] = False
                    final_selected = returned_df.index[returned_df["Selected"]].tolist()
                    st.session_state["selected_msg_id"] = final_selected[0] if final_selected else None

        with col_right:
            st.subheader("Message Analysis Drilldown")
            selected_id = st.session_state.get("selected_msg_id", None)
            if selected_id is not None:
                selection_row = filtered_df.loc[filtered_df["message_id"] == selected_id]
                if not selection_row.empty:
                    show_message_details(selection_row.iloc[0])
                else:
                    st.info("The selected message is not in the filtered dataset.")
            else:
                st.info("Select a row from the left table to view details here.")

    # ------------------
    # Competitor Mentions
    # ------------------
    with tabs[1]:
        df_all = get_data()
        show_category_tab_as_cards(df_all, "competitor_mentions", "Competitor Mentions")

    # ------------------
    # Product Praise
    # ------------------
    with tabs[2]:
        df_all = get_data()
        show_category_tab_as_cards(df_all, "product_praise", "Product Praise")

    # ------------------
    # Product Complaints
    # ------------------
    with tabs[3]:
        df_all = get_data()
        show_category_tab_as_cards(df_all, "product_complaints", "Product Complaints")

    # ------------------
    # Product Technical Problems
    # ------------------
    with tabs[4]:
        df_all = get_data()
        show_category_tab_as_cards(df_all, "product_technical_problem", "Product Technical Problems")

    # ------------------
    # Feature Requests
    # ------------------
    with tabs[5]:
        df_all = get_data()
        show_category_tab_as_cards(df_all, "feature_requests", "Feature Requests")

    # ------------------
    # Product Justification
    # ------------------
    with tabs[6]:
        df_all = get_data()
        show_category_tab_as_cards(df_all, "product_justification", "Product Justification")


if __name__ == "__main__":
    main()