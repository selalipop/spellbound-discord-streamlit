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
    if author_filter.strip():
        df = df[df["author_username"].str.contains(author_filter, case=False, na=False)]
    if content_filter.strip():
        df = df[df["content"].str.contains(content_filter, case=False, na=False)]
    if channel_filter.strip():
        df = df[df["channel_name"].str.contains(channel_filter, case=False, na=False)]
    df = df[df["sentiment_score"].between(sentiment_min, sentiment_max)]
    df = df[df["relevance_score"].between(relevance_min, relevance_max)]
    return df

# ------------------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Message Analysis Explorer", layout="wide")
    tab_main, tab_other = st.tabs(["Main Analysis Explorer", "Another Tab"])
    
    with tab_main:
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

        # Filter
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
                # Clear selection
                st.session_state["selected_msg_id"] = None
            else:
                # Prepare data for st.data_editor
                # STEP 1: Initialize st.session_state data if needed
                if "table_source" not in st.session_state:
                    st.session_state["table_source"] = pd.DataFrame()
                if "selected_msg_id" not in st.session_state:
                    st.session_state["selected_msg_id"] = None
                
                # Build a fresh table each run
                table_df = filtered_df.copy()
                table_df["Truncated"] = table_df["content"].apply(
                    lambda x: x.replace("\n"," ")[:80] + ("..." if len(x) > 80 else "")
                )
                table_df["Selected"] = False

                # Restore existing selection if it is still in filtered_df
                if st.session_state["selected_msg_id"] in table_df["message_id"].values:
                    idx = table_df[table_df["message_id"] == st.session_state["selected_msg_id"]].index
                    table_df.loc[idx, "Selected"] = True

                # We’ll hide everything except "Selected"/"Truncated"
                # but keep message_id in the index so we can identify rows
                table_df.set_index("message_id", inplace=True, drop=False)

                # STEP 2: Show st.data_editor 
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
                            disabled=True  # read-only
                        )
                    },
                    num_rows="fixed",
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                    key="my_data_editor"
                )

                # STEP 3: Single-row selection logic 
                # Compare returned_df['Selected'] with the previous state 
                # so that only one row can remain checked
                if not returned_df.equals(table_df[["Selected","Truncated"]]):
                    # The user changed something in the checkboxes
                    # Determine which rows are now checked
                    selected_indexes = returned_df.index[returned_df["Selected"] == True].tolist()
                    if len(selected_indexes) > 1:
                        # More than one row is selected, keep the last one
                        last_idx = selected_indexes[-1]
                        for i in selected_indexes:
                            if i != last_idx:
                                returned_df.at[i, "Selected"] = False
                        # Re-run so that the data_editor is updated
                    # Now there's 0 or 1 row selected
                    final_selected = returned_df.index[returned_df["Selected"]].tolist()
                    st.session_state["selected_msg_id"] = final_selected[0] if final_selected else None

                # If the user unchecks the row, then final_selected is empty, we set st.session_state to None

        with col_right:
            st.subheader("Message Analysis Drilldown")
            selected_id = st.session_state.get("selected_msg_id", None)
            if selected_id is not None:
                selection_row = filtered_df.loc[filtered_df["message_id"] == selected_id]
                if not selection_row.empty:
                    row = selection_row.iloc[0]
                    
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
                        f" Channel: {row['channel_name']} &nbsp;|&nbsp;"
                        f" Time: {rel_time} &nbsp;&nbsp;"
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

                    # ADDITIONAL DETAILS
                    st.markdown("### Additional Details")
                    def display_list_in_markdown(label, items):
                        if items:
                            # Might be list or single string
                            actual_list = items if isinstance(items, list) else [items]
                            if len(actual_list) > 0:
                                st.markdown(f"**{label}:**")
                                for val in actual_list:
                                    st.markdown(f"- {val}")

                    display_list_in_markdown("Competitor Mentions", row["competitor_mentions"])
                    display_list_in_markdown("Product Praise", row["product_praise"])
                    display_list_in_markdown("Product Complaints", row["product_complaints"])
                    display_list_in_markdown("Product Technical Problems", row["product_technical_problem"])
                    display_list_in_markdown("Feature Requests", row["feature_requests"])
                    display_list_in_markdown("Product Justification", row["product_justification"])
                else:
                    st.info("The selected message is not in the filtered dataset.")
            else:
                st.info("Select a row from the left table to view details here.")

    with tab_other:
        st.write("Another tab for future expansions.")


if __name__ == "__main__":
    main()
