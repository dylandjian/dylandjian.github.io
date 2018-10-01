import React from "react"

// Import typefaces
import "typeface-montserrat"
import "typeface-merriweather"

import profilePic from "./profile-pic.jpg"

import githubLogo from "./github-logo.svg"
import linkedinLogo from "./linkedin-logo.svg"
import twitterLogo from "./twitter-logo.svg"
import twitchLogo from "./twitch-logo.png"

import { OutboundLink } from "gatsby-plugin-google-analytics"
import { rhythm } from "../utils/typography"

class Bio extends React.Component {
	render() {
		return (
			<div
				style={{
					display: "flex",
					flexDirection: "column"
				}}
			>
				<div
					style={{
						display: "flex",
						marginTop: "15px"
					}}
				>
					<img
						src={profilePic}
						alt={`Dylan Djian`}
						style={{
							marginRight: rhythm(1 / 2),
							marginBottom: 0,
							borderRadius: "5px",
							width: rhythm(2.6),
							height: rhythm(2.6)
						}}
					/>
					<p
						style={{
							marginTop: "-6px"
						}}
					>
						Welcome to my blog ! I am a software developer student{" "}
						<a href="http://www.42.fr/">@42 Paris</a>, self-taught and
            passionate about machine learning and interested in
						new web technologies. <br />
					</p>
				</div>
				<div
					style={{
						display: "flex",
						justifyContent: "center",
						marginBottom: "20px"
					}}
				>
					<div>
						<a
							href="https://www.linkedin.com/in/dylan-djian/"
							style={{
								width: "0px",
								height: "0px",
								boxShadow: "none"
							}}
						>
							<img
								src={linkedinLogo}
								alt={`LinkedIn`}
								style={{
									width: rhythm(1),
									height: rhythm(1),
									outline: "none",
									marginBottom: 0
								}}
							/>
						</a>
					</div>
					<div
						style={{
							marginLeft: rhythm(0.7)
						}}
					>
						<a
							href="https://github.com/dylandjian"
							style={{
								width: "0px",
								height: "0px",
								boxShadow: "none"
							}}
						>
							<img
								src={githubLogo}
								alt={`Github`}
								style={{
									width: rhythm(1),
									height: rhythm(1),
									marginBottom: 0,
									marginTop: "2px"
								}}
							/>
						</a>
					</div>
					<div
						style={{
							marginLeft: rhythm(0.7)
						}}
					>
						<a
							href="https://twitter.com/dylandjian"
							style={{
								width: "0px",
								height: "0px",
								boxShadow: "none"
							}}
						>
							<img
								src={twitterLogo}
								alt={`Twitter`}
								style={{
									width: rhythm(1),
									height: rhythm(1),
									marginBottom: 0,
									marginTop: "2px"
								}}
							/>
						</a>
					</div>
					<div
						style={{
							marginLeft: rhythm(0.7)
						}}
					>
						<a
							href="https://twitch.tv/sulozor"
							style={{
								width: "0px",
								height: "0px",
								boxShadow: "none"
							}}
						>
							<img
								src={twitchLogo}
								alt={`Twitch`}
								style={{
									width: rhythm(1),
									height: rhythm(1),
									marginBottom: 0,
									marginTop: "2px"
								}}
							/>
						</a>
					</div>
				</div>
			</div>
		)
	}
}

export default Bio
